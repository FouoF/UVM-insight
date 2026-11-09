// Package kube maps pod UIDs reported by the collector to pod names using the
// in-cluster service account. It lists only the pods on this node.
package kube

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"sync"
	"time"
)

const saDir = "/var/run/secrets/kubernetes.io/serviceaccount"

type Pod struct {
	Namespace string
	Name      string
}

type Resolver struct {
	client *http.Client
	base   string
	node   string

	mu       sync.Mutex
	pods     map[string]Pod
	lastList time.Time
	kick     chan struct{}
}

// InCluster returns nil, nil when the agent is not running in a pod.
func InCluster(node string) (*Resolver, error) {
	host, port := os.Getenv("KUBERNETES_SERVICE_HOST"), os.Getenv("KUBERNETES_SERVICE_PORT")
	if host == "" || port == "" {
		return nil, nil
	}
	if node == "" {
		return nil, fmt.Errorf("node name is empty; set NODE_NAME from spec.nodeName")
	}
	ca, err := os.ReadFile(saDir + "/ca.crt")
	if err != nil {
		return nil, err
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(ca) {
		return nil, fmt.Errorf("no certificates in %s/ca.crt", saDir)
	}
	return &Resolver{
		client: &http.Client{
			Timeout:   10 * time.Second,
			Transport: &http.Transport{TLSClientConfig: &tls.Config{RootCAs: pool}},
		},
		base: "https://" + net.JoinHostPort(host, port),
		node: node,
		pods: map[string]Pod{},
		kick: make(chan struct{}, 1),
	}, nil
}

// Lookup never blocks. A miss schedules a refresh, so a pod that started
// after the last list is resolved on a later scrape.
func (r *Resolver) Lookup(uid string) (Pod, bool) {
	r.mu.Lock()
	p, ok := r.pods[uid]
	r.mu.Unlock()
	if !ok {
		select {
		case r.kick <- struct{}{}:
		default:
		}
	}
	return p, ok
}

func (r *Resolver) Run(ctx context.Context) {
	r.refresh(ctx)
	tick := time.NewTicker(time.Minute)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
			r.refresh(ctx)
		case <-r.kick:
			r.mu.Lock()
			recent := time.Since(r.lastList) < 5*time.Second
			r.mu.Unlock()
			if !recent {
				r.refresh(ctx)
			}
		}
	}
}

type podList struct {
	Items []struct {
		Metadata struct {
			UID       string `json:"uid"`
			Name      string `json:"name"`
			Namespace string `json:"namespace"`
		} `json:"metadata"`
	} `json:"items"`
}

func (r *Resolver) refresh(ctx context.Context) {
	r.mu.Lock()
	r.lastList = time.Now()
	r.mu.Unlock()

	// Projected tokens rotate, so read the file on every request.
	token, err := os.ReadFile(saDir + "/token")
	if err != nil {
		log.Printf("kube: %v", err)
		return
	}
	q := url.Values{"fieldSelector": {"spec.nodeName=" + r.node}}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, r.base+"/api/v1/pods?"+q.Encode(), nil)
	if err != nil {
		return
	}
	req.Header.Set("Authorization", "Bearer "+string(token))
	req.Header.Set("Accept", "application/json")
	resp, err := r.client.Do(req)
	if err != nil {
		log.Printf("kube: list pods: %v", err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		log.Printf("kube: list pods: %s", resp.Status)
		return
	}
	var list podList
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		log.Printf("kube: decode pods: %v", err)
		return
	}
	pods := make(map[string]Pod, len(list.Items))
	for _, it := range list.Items {
		pods[it.Metadata.UID] = Pod{Namespace: it.Metadata.Namespace, Name: it.Metadata.Name}
	}
	r.mu.Lock()
	r.pods = pods
	r.mu.Unlock()
}
