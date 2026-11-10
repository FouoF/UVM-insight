package main

import (
	"context"
	"flag"
	"io/fs"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/FouoF/UVM-insight/agent/pkg/aggregate"
	"github.com/FouoF/UVM-insight/agent/pkg/api"
	"github.com/FouoF/UVM-insight/agent/pkg/ingest"
	"github.com/FouoF/UVM-insight/agent/pkg/kube"
	"github.com/FouoF/UVM-insight/agent/pkg/mock"
	"github.com/FouoF/UVM-insight/agent/pkg/webui"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "install" {
		runInstall(os.Args[2:])
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "webhook" {
		runWebhook(os.Args[2:])
		return
	}
	socketPath := flag.String("socket", "/tmp/uvm-insight/agent.sock", "Unix socket the CUPTI collector connects to")
	listen := flag.String("listen", "127.0.0.1:9400", "HTTP address for the UI, API, and /metrics")
	retention := flag.Duration("retention", 10*time.Minute, "how long to keep samples and exited processes")
	useMock := flag.Bool("mock", false, "emit a demo workload instead of waiting for a CUDA process")
	webDir := flag.String("web-dir", "", "serve this directory instead of the UI embedded in the binary")
	ui := flag.Bool("ui", true, "serve the web UI and JSON API; with false only /metrics and /healthz are served")
	nodeName := flag.String("node-name", os.Getenv("NODE_NAME"), "Kubernetes node this agent runs on, used to resolve pod names")
	flag.Parse()

	store := aggregate.New(*retention)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	pods, err := kube.InCluster(*nodeName)
	switch {
	case err != nil:
		log.Printf("pod names unavailable: %v", err)
	case pods != nil:
		go pods.Run(ctx)
		store.SetPodLookup(func(uid string) (string, string, bool) {
			p, ok := pods.Lookup(uid)
			return p.Namespace, p.Name, ok
		})
		log.Printf("resolving pods on node %s", *nodeName)
	}

	if *useMock {
		mock.Run(ctx, store)
		log.Printf("mock workload enabled")
	}

	go func() {
		ticker := time.NewTicker(15 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				store.Expire()
			}
		}
	}()

	go func() {
		if err := ingest.Serve(ctx, *socketPath, store); err != nil {
			log.Fatalf("socket: %v", err)
		}
	}()

	handler := api.MetricsHandler(store)
	if *ui {
		var web fs.FS = webui.Dist()
		if *webDir != "" {
			web = os.DirFS(*webDir)
		}
		handler = api.Handler(store, web)
	}
	srv := &http.Server{
		Addr:              *listen,
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdown)
	}()

	if *ui {
		log.Printf("ui and metrics on http://%s", *listen)
	} else {
		log.Printf("metrics on http://%s/metrics", *listen)
	}
	log.Printf("collector socket %s", *socketPath)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}
