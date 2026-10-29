package api

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/FouoF/UVM-insight/agent/pkg/aggregate"
	"github.com/FouoF/UVM-insight/agent/pkg/protocol"
)

func TestProcessesAndMetrics(t *testing.T) {
	st := aggregate.New(time.Minute)
	now := time.Unix(1_700_000_100, 500*int64(time.Millisecond))
	st.SetClock(func() time.Time { return now })
	st.Register(protocol.Hello{PID: 3, AppName: "demo"})
	st.IngestAt(3, protocol.EventBatch{Uvm: []protocol.UvmEvent{{
		Kind: protocol.KindGPUPageFault, Bytes: 2, DeviceID: 0, SrcID: 0,
	}}}, now)

	srv := httptest.NewServer(Handler(st, nil))
	defer srv.Close()

	res, err := http.Get(srv.URL + "/api/processes")
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	var procs []aggregate.ProcessInfo
	if err := json.NewDecoder(res.Body).Decode(&procs); err != nil {
		t.Fatal(err)
	}
	if len(procs) != 1 || procs[0].AppName != "demo" {
		t.Fatalf("%+v", procs)
	}

	mres, err := http.Get(srv.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer mres.Body.Close()
	body, _ := io.ReadAll(mres.Body)
	if !strings.Contains(string(body), "uvm_page_faults_total") {
		t.Fatalf("metrics:\n%s", body)
	}
}
