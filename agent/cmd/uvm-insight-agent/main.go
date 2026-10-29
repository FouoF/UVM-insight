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
	"github.com/FouoF/UVM-insight/agent/pkg/mock"
	"github.com/FouoF/UVM-insight/agent/pkg/webui"
)

func main() {
	socketPath := flag.String("socket", "/tmp/uvm-insight/agent.sock", "Unix socket the CUPTI collector connects to")
	listen := flag.String("listen", "127.0.0.1:9400", "HTTP address for the UI, API, and /metrics")
	retention := flag.Duration("retention", 10*time.Minute, "how long to keep samples and exited processes")
	useMock := flag.Bool("mock", false, "emit a demo workload instead of waiting for a CUDA process")
	webDir := flag.String("web-dir", "", "serve this directory instead of the UI embedded in the binary")
	flag.Parse()

	store := aggregate.New(*retention)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

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

	var web fs.FS = webui.Dist()
	if *webDir != "" {
		web = os.DirFS(*webDir)
	}
	srv := &http.Server{
		Addr:              *listen,
		Handler:           api.Handler(store, web),
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdown)
	}()

	log.Printf("ui and metrics on http://%s", *listen)
	log.Printf("collector socket %s", *socketPath)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}
