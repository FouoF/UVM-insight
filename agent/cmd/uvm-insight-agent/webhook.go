package main

import (
	"flag"
	"log"
	"strings"

	"github.com/FouoF/UVM-insight/agent/pkg/webhook"
)

func runWebhook(args []string) {
	fs := flag.NewFlagSet("webhook", flag.ExitOnError)
	listen := fs.String("listen", ":9443", "HTTPS listen address")
	certFile := fs.String("tls-cert", "/certs/tls.crt", "TLS certificate")
	keyFile := fs.String("tls-key", "/certs/tls.key", "TLS private key")
	hostDir := fs.String("host-dir", "/var/lib/uvm-insight", "host path where the DaemonSet installs the collector")
	mountPath := fs.String("mount-path", "", "path inside the workload container (default: same as host-dir)")
	addSysAdmin := fs.Bool("add-sys-admin", true, "add SYS_ADMIN so CUPTI can enable unified-memory counters")
	injectHAMi := fs.Bool("inject-hami", true, "inject when the pod has the nvidia.com/use-gpuuuid annotation")
	gpuResources := fs.String("gpu-resources", "nvidia.com/gpu", "comma-separated resource names that trigger injection")
	ownNS := fs.String("own-namespace", "", "skip pods in this namespace (the exporter's namespace)")
	ignoreNS := fs.String("ignore-namespaces", "kube-system,kube-public,kube-node-lease", "comma-separated namespaces to skip")
	_ = fs.Parse(args)

	ignore := map[string]struct{}{}
	for _, ns := range strings.Split(*ignoreNS, ",") {
		ns = strings.TrimSpace(ns)
		if ns != "" {
			ignore[ns] = struct{}{}
		}
	}
	var resources []string
	for _, r := range strings.Split(*gpuResources, ",") {
		r = strings.TrimSpace(r)
		if r != "" {
			resources = append(resources, r)
		}
	}

	srv := &webhook.Server{
		Config: webhook.Config{
			HostDir:      *hostDir,
			MountPath:    *mountPath,
			AddSysAdmin:  *addSysAdmin,
			GPUResources: resources,
			InjectOnHAMi: *injectHAMi,
			IgnoreNS:     ignore,
			OwnNamespace: *ownNS,
		},
		Log: log.Default(),
	}
	log.Printf("mutating webhook on https://%s", *listen)
	log.Fatal(srv.ListenAndServeTLS(*listen, *certFile, *keyFile))
}
