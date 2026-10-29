package api

import (
	"encoding/json"
	"io/fs"
	"net/http"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/FouoF/UVM-insight/agent/pkg/aggregate"
	"github.com/FouoF/UVM-insight/agent/pkg/metrics"
)

func Handler(st *aggregate.Store, web fs.FS) http.Handler {
	mux := http.NewServeMux()
	reg := prometheus.NewRegistry()
	reg.MustRegister(metrics.New(st))
	mux.Handle("GET /metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{}))
	mux.HandleFunc("GET /api/processes", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, st.Processes())
	})
	mux.HandleFunc("GET /api/snapshot", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, st.Snapshot(pidQuery(r)))
	})
	mux.HandleFunc("GET /api/heatmap", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, st.Heatmap(pidQuery(r)))
	})
	mux.HandleFunc("GET /api/stream", func(w http.ResponseWriter, r *http.Request) {
		stream(st, w, r)
	})
	if web != nil {
		mux.Handle("/", spa(web))
	}
	return mux
}

func pidQuery(r *http.Request) uint32 {
	q := r.URL.Query().Get("pid")
	if q == "" || q == "all" {
		return 0
	}
	n, err := strconv.ParseUint(q, 10, 32)
	if err != nil {
		return 0
	}
	return uint32(n)
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}

func stream(st *aggregate.Store, w http.ResponseWriter, r *http.Request) {
	conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{OriginPatterns: []string{"*"}})
	if err != nil {
		return
	}
	defer conn.Close(websocket.StatusNormalClosure, "")
	ctx := conn.CloseRead(r.Context())
	pid := pidQuery(r)
	var since uint64
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	send := func() error {
		msg := st.Stream(pid, since)
		since = msg.LastSeq
		return wsjson.Write(ctx, conn, msg)
	}
	if err := send(); err != nil {
		return
	}
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if err := send(); err != nil {
				return
			}
		}
	}
}

func spa(fsys fs.FS) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			http.NotFound(w, r)
			return
		}
		name := strings.TrimPrefix(path.Clean("/"+r.URL.Path), "/")
		if name == "" || name == "." {
			name = "index.html"
		}
		if _, err := fs.Stat(fsys, name); err != nil {
			name = "index.html"
		}
		http.ServeFileFS(w, r, fsys, name)
	})
}
