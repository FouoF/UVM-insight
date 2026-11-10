package webhook

import (
	"crypto/tls"
	"encoding/json"
	"io"
	"log"
	"net/http"
)

// AdmissionReview is the v1 admission.k8s.io payload we accept and return.
type AdmissionReview struct {
	APIVersion string             `json:"apiVersion"`
	Kind       string             `json:"kind"`
	Request    *AdmissionRequest  `json:"request,omitempty"`
	Response   *AdmissionResponse `json:"response,omitempty"`
}

type AdmissionRequest struct {
	UID       string          `json:"uid"`
	Namespace string          `json:"namespace"`
	Name      string          `json:"name"`
	Operation string          `json:"operation"`
	Object    json.RawMessage `json:"object"`
}

type AdmissionResponse struct {
	UID       string  `json:"uid"`
	Allowed   bool    `json:"allowed"`
	Patch     []byte  `json:"patch,omitempty"`
	PatchType *string `json:"patchType,omitempty"`
	Result    *Status `json:"result,omitempty"`
}

type Status struct {
	Message string `json:"message,omitempty"`
}

// Server serves the mutating webhook over HTTPS.
type Server struct {
	Config Config
	Log    *log.Logger
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("ok\n"))
	})
	mux.HandleFunc("/mutate", s.mutate)
	return mux
}

func (s *Server) mutate(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 8<<20))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	var review AdmissionReview
	if err := json.Unmarshal(body, &review); err != nil || review.Request == nil {
		http.Error(w, "invalid AdmissionReview", http.StatusBadRequest)
		return
	}
	resp := s.decide(review.Request)
	out := AdmissionReview{
		APIVersion: review.APIVersion,
		Kind:       review.Kind,
		Response:   resp,
	}
	if out.APIVersion == "" {
		out.APIVersion = "admission.k8s.io/v1"
	}
	if out.Kind == "" {
		out.Kind = "AdmissionReview"
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

func (s *Server) decide(req *AdmissionRequest) *AdmissionResponse {
	resp := &AdmissionResponse{UID: req.UID, Allowed: true}
	if req.Operation != "CREATE" && req.Operation != "UPDATE" {
		return resp
	}
	var pod Pod
	if err := json.Unmarshal(req.Object, &pod); err != nil {
		resp.Result = &Status{Message: "decode pod: " + err.Error()}
		return resp
	}
	if pod.Metadata.Namespace == "" {
		pod.Metadata.Namespace = req.Namespace
	}
	patch, err := s.Config.Patch(&pod)
	if err != nil {
		s.logf("patch %s/%s: %v", pod.Metadata.Namespace, pod.Metadata.Name, err)
		resp.Result = &Status{Message: err.Error()}
		return resp
	}
	if len(patch) == 0 {
		return resp
	}
	pt := "JSONPatch"
	resp.Patch = patch
	resp.PatchType = &pt
	s.logf("%s/%s %s", pod.Metadata.Namespace, nameOf(&pod), s.Config.Explain(&pod))
	return resp
}

func nameOf(pod *Pod) string {
	if pod.Metadata.Name != "" {
		return pod.Metadata.Name
	}
	return pod.Metadata.GenerateName + "*"
}

func (s *Server) logf(format string, args ...any) {
	if s.Log != nil {
		s.Log.Printf(format, args...)
		return
	}
	log.Printf(format, args...)
}

// ListenAndServeTLS serves HTTPS with the given certificate files.
func (s *Server) ListenAndServeTLS(addr, certFile, keyFile string) error {
	srv := &http.Server{
		Addr:      addr,
		Handler:   s.Handler(),
		TLSConfig: &tls.Config{MinVersion: tls.VersionTLS12},
	}
	return srv.ListenAndServeTLS(certFile, keyFile)
}
