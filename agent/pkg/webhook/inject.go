// Package webhook mutates Pods so CUDA workloads pick up the CUPTI collector.
package webhook

import (
	"encoding/json"
	"fmt"
	"path"
	"strconv"
	"strings"
)

const (
	AnnotInject     = "uvm-insight.dynamia.ai/inject"
	AnnotInjected   = "uvm-insight.dynamia.ai/injected"
	AnnotUseGPUUUID = "nvidia.com/use-gpuuuid"
	VolumeName      = "uvm-insight"
	EnvInjection    = "CUDA_INJECTION64_PATH"
	EnvSocket       = "UVMI_SOCKET"
	EnvAppName      = "UVMI_APP_NAME"
)

// Config controls which Pods get the collector and how it is mounted.
type Config struct {
	HostDir      string
	MountPath    string
	AddSysAdmin  bool
	GPUResources []string
	InjectOnHAMi bool // nvidia.com/use-gpuuuid annotation
	IgnoreNS     map[string]struct{}
	OwnNamespace string
}

func (c Config) mountPath() string {
	if c.MountPath != "" {
		return c.MountPath
	}
	if c.HostDir != "" {
		return c.HostDir
	}
	return "/var/lib/uvm-insight"
}

func (c Config) hostDir() string {
	if c.HostDir != "" {
		return c.HostDir
	}
	return "/var/lib/uvm-insight"
}

// Pod is the subset of corev1.Pod needed to decide and build a patch.
type Pod struct {
	Metadata ObjectMetadata `json:"metadata"`
	Spec     Spec           `json:"spec"`
}

type ObjectMetadata struct {
	Name         string            `json:"name"`
	Namespace    string            `json:"namespace"`
	GenerateName string            `json:"generateName"`
	Labels       map[string]string `json:"labels"`
	Annotations  map[string]string `json:"annotations"`
}

type Spec struct {
	Containers []Container `json:"containers"`
	Volumes    []Volume    `json:"volumes"`
}

type Container struct {
	Name            string           `json:"name"`
	Env             []EnvVar         `json:"env"`
	Resources       Resources        `json:"resources"`
	VolumeMounts    []VolumeMount    `json:"volumeMounts"`
	SecurityContext *SecurityContext `json:"securityContext"`
}

type EnvVar struct {
	Name      string `json:"name"`
	Value     string `json:"value,omitempty"`
	ValueFrom any    `json:"valueFrom,omitempty"`
}

type Resources struct {
	Limits   map[string]any `json:"limits"`
	Requests map[string]any `json:"requests"`
}

type VolumeMount struct {
	Name      string `json:"name"`
	MountPath string `json:"mountPath"`
	ReadOnly  bool   `json:"readOnly,omitempty"`
}

type Volume struct {
	Name     string    `json:"name"`
	HostPath *HostPath `json:"hostPath,omitempty"`
}

type HostPath struct {
	Path string `json:"path"`
	Type string `json:"type,omitempty"`
}

type SecurityContext struct {
	Capabilities *Capabilities `json:"capabilities,omitempty"`
}

type Capabilities struct {
	Add  []string `json:"add,omitempty"`
	Drop []string `json:"drop,omitempty"`
}

type patchOp struct {
	Op    string `json:"op"`
	Path  string `json:"path"`
	Value any    `json:"value,omitempty"`
}

// Patch returns a JSON patch that injects the collector, or nil when the Pod
// should be left alone.
func (c Config) Patch(pod *Pod) ([]byte, error) {
	if pod == nil {
		return nil, nil
	}
	if _, skip := c.IgnoreNS[pod.Metadata.Namespace]; skip {
		return nil, nil
	}
	if c.OwnNamespace != "" && pod.Metadata.Namespace == c.OwnNamespace {
		return nil, nil
	}
	ann := pod.Metadata.Annotations
	if ann != nil {
		if v := strings.ToLower(strings.TrimSpace(ann[AnnotInject])); v == "false" || v == "0" || v == "no" {
			return nil, nil
		}
		if ann[AnnotInjected] == "true" {
			return nil, nil
		}
	}
	idxs := c.targetContainers(pod)
	if len(idxs) == 0 {
		return nil, nil
	}

	var ops []patchOp
	mp := c.mountPath()
	hd := c.hostDir()
	lib := path.Join(mp, "lib", "libuvminsight.so")
	sock := path.Join(mp, "agent.sock")

	if pod.Metadata.Annotations == nil {
		ops = append(ops, patchOp{Op: "add", Path: "/metadata/annotations", Value: map[string]string{}})
	}
	ops = append(ops, patchOp{Op: "add", Path: "/metadata/annotations/" + escapeJSONPointer(AnnotInjected), Value: "true"})

	if !hasVolume(pod.Spec.Volumes, VolumeName) {
		vol := Volume{Name: VolumeName, HostPath: &HostPath{Path: hd, Type: "DirectoryOrCreate"}}
		if pod.Spec.Volumes == nil {
			ops = append(ops, patchOp{Op: "add", Path: "/spec/volumes", Value: []Volume{vol}})
		} else {
			ops = append(ops, patchOp{Op: "add", Path: "/spec/volumes/-", Value: vol})
		}
	}

	app := pod.Metadata.Name
	if app == "" {
		app = strings.TrimSuffix(pod.Metadata.GenerateName, "-")
	}

	for _, i := range idxs {
		base := "/spec/containers/" + strconv.Itoa(i)
		ctr := pod.Spec.Containers[i]
		want := []EnvVar{
			{Name: EnvInjection, Value: lib},
			{Name: EnvSocket, Value: sock},
		}
		if !hasEnv(ctr.Env, EnvAppName) && app != "" {
			want = append(want, EnvVar{Name: EnvAppName, Value: app})
		}
		ops = append(ops, ensureEnvs(base, ctr.Env, want)...)
		if !hasMount(ctr.VolumeMounts, VolumeName) {
			vm := VolumeMount{Name: VolumeName, MountPath: mp, ReadOnly: true}
			if ctr.VolumeMounts == nil {
				ops = append(ops, patchOp{Op: "add", Path: base + "/volumeMounts", Value: []VolumeMount{vm}})
			} else {
				ops = append(ops, patchOp{Op: "add", Path: base + "/volumeMounts/-", Value: vm})
			}
		}
		if c.AddSysAdmin {
			ops = append(ops, ensureSysAdmin(base, ctr.SecurityContext)...)
		}
	}

	if len(ops) == 0 {
		return nil, nil
	}
	return json.Marshal(ops)
}

func (c Config) targetContainers(pod *Pod) []int {
	ann := pod.Metadata.Annotations
	force := false
	if ann != nil {
		v := strings.ToLower(strings.TrimSpace(ann[AnnotInject]))
		force = v == "true" || v == "1" || v == "yes"
		if c.InjectOnHAMi && ann[AnnotUseGPUUUID] != "" {
			force = true
		}
	}
	var idxs []int
	for i, ctr := range pod.Spec.Containers {
		if force || c.containerRequestsGPU(ctr) {
			idxs = append(idxs, i)
		}
	}
	return idxs
}

func (c Config) containerRequestsGPU(ctr Container) bool {
	resources := []map[string]any{ctr.Resources.Limits, ctr.Resources.Requests}
	for _, res := range resources {
		for _, name := range c.GPUResources {
			if _, ok := res[name]; ok {
				return true
			}
		}
	}
	return false
}

func ensureEnvs(base string, existing []EnvVar, want []EnvVar) []patchOp {
	var add []EnvVar
	for _, ev := range want {
		if !hasEnv(existing, ev.Name) {
			add = append(add, ev)
		}
	}
	if len(add) == 0 {
		return nil
	}
	if existing == nil {
		return []patchOp{{Op: "add", Path: base + "/env", Value: add}}
	}
	ops := make([]patchOp, 0, len(add))
	for _, ev := range add {
		ops = append(ops, patchOp{Op: "add", Path: base + "/env/-", Value: ev})
	}
	return ops
}

func ensureSysAdmin(base string, sc *SecurityContext) []patchOp {
	if sc == nil {
		return []patchOp{{
			Op:    "add",
			Path:  base + "/securityContext",
			Value: SecurityContext{Capabilities: &Capabilities{Add: []string{"SYS_ADMIN"}}},
		}}
	}
	if sc.Capabilities == nil {
		return []patchOp{{
			Op:    "add",
			Path:  base + "/securityContext/capabilities",
			Value: Capabilities{Add: []string{"SYS_ADMIN"}},
		}}
	}
	for _, a := range sc.Capabilities.Add {
		if a == "SYS_ADMIN" {
			return nil
		}
	}
	return []patchOp{{Op: "add", Path: base + "/securityContext/capabilities/add/-", Value: "SYS_ADMIN"}}
}

func hasEnv(env []EnvVar, name string) bool {
	for _, e := range env {
		if e.Name == name {
			return true
		}
	}
	return false
}

func hasMount(ms []VolumeMount, name string) bool {
	for _, m := range ms {
		if m.Name == name {
			return true
		}
	}
	return false
}

func hasVolume(vs []Volume, name string) bool {
	for _, v := range vs {
		if v.Name == name {
			return true
		}
	}
	return false
}

func escapeJSONPointer(s string) string {
	s = strings.ReplaceAll(s, "~", "~0")
	s = strings.ReplaceAll(s, "/", "~1")
	return s
}

// Explain returns a short reason used in logs.
func (c Config) Explain(pod *Pod) string {
	if pod == nil {
		return "nil"
	}
	idxs := c.targetContainers(pod)
	if len(idxs) == 0 {
		return "skip"
	}
	names := make([]string, 0, len(idxs))
	for _, i := range idxs {
		names = append(names, pod.Spec.Containers[i].Name)
	}
	return fmt.Sprintf("inject %s", strings.Join(names, ","))
}
