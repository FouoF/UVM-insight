package webhook

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestPatchGPUContainer(t *testing.T) {
	cfg := Config{
		HostDir:      "/var/lib/uvm-insight",
		AddSysAdmin:  true,
		GPUResources: []string{"nvidia.com/gpu"},
		InjectOnHAMi: true,
	}
	pod := &Pod{
		Metadata: ObjectMetadata{Name: "esm2-abc", Namespace: "pharma"},
		Spec: Spec{
			Containers: []Container{{
				Name:      "model",
				Resources: Resources{Limits: map[string]any{"nvidia.com/gpu": "1"}},
			}},
		},
	}
	raw, err := cfg.Patch(pod)
	if err != nil || raw == nil {
		t.Fatalf("patch: %v %s", err, raw)
	}
	var ops []patchOp
	if err := json.Unmarshal(raw, &ops); err != nil {
		t.Fatal(err)
	}
	joined, _ := json.Marshal(ops)
	s := string(joined)
	for _, want := range []string{
		`CUDA_INJECTION64_PATH`,
		`/var/lib/uvm-insight/lib/libuvminsight.so`,
		`UVMI_SOCKET`,
		`SYS_ADMIN`,
		`uvm-insight.dynamia.ai~1injected`,
		`"name":"uvm-insight"`,
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %q in %s", want, s)
		}
	}
}

func TestSkipOptOut(t *testing.T) {
	cfg := Config{GPUResources: []string{"nvidia.com/gpu"}, InjectOnHAMi: true}
	pod := &Pod{
		Metadata: ObjectMetadata{
			Annotations: map[string]string{AnnotInject: "false"},
		},
		Spec: Spec{Containers: []Container{{
			Name:      "model",
			Resources: Resources{Limits: map[string]any{"nvidia.com/gpu": "1"}},
		}}},
	}
	raw, err := cfg.Patch(pod)
	if err != nil || raw != nil {
		t.Fatalf("got %s err=%v", raw, err)
	}
}

func TestInjectOnHAMiAnnotation(t *testing.T) {
	cfg := Config{HostDir: "/var/lib/uvm-insight", InjectOnHAMi: true, AddSysAdmin: true}
	pod := &Pod{
		Metadata: ObjectMetadata{
			Name:        "biogpt-1",
			Annotations: map[string]string{AnnotUseGPUUUID: "GPU-abc"},
		},
		Spec: Spec{Containers: []Container{{Name: "model"}}},
	}
	raw, err := cfg.Patch(pod)
	if err != nil || raw == nil {
		t.Fatalf("patch: %v %s", err, raw)
	}
}

func TestIdempotentWhenAlreadyInjected(t *testing.T) {
	cfg := Config{HostDir: "/var/lib/uvm-insight", GPUResources: []string{"nvidia.com/gpu"}, AddSysAdmin: true}
	pod := &Pod{
		Metadata: ObjectMetadata{
			Name:        "x",
			Annotations: map[string]string{AnnotInjected: "true"},
		},
		Spec: Spec{Containers: []Container{{
			Name:      "model",
			Resources: Resources{Limits: map[string]any{"nvidia.com/gpu": "1"}},
		}}},
	}
	raw, err := cfg.Patch(pod)
	if err != nil || raw != nil {
		t.Fatalf("got %s err=%v", raw, err)
	}
}

func TestIgnoreOwnNamespace(t *testing.T) {
	cfg := Config{OwnNamespace: "uvm-insight", GPUResources: []string{"nvidia.com/gpu"}}
	pod := &Pod{
		Metadata: ObjectMetadata{Namespace: "uvm-insight"},
		Spec: Spec{Containers: []Container{{
			Name:      "exporter",
			Resources: Resources{Limits: map[string]any{"nvidia.com/gpu": "1"}},
		}}},
	}
	raw, err := cfg.Patch(pod)
	if err != nil || raw != nil {
		t.Fatalf("got %s err=%v", raw, err)
	}
}
