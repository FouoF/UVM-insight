package main

import (
	"bytes"
	"crypto/sha256"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
)

// runInstall copies the collector libraries to a host directory that CUDA
// workloads mount. Running processes keep the old files mapped, so each file
// is written to a temporary name and renamed into place instead of being
// overwritten.
func runInstall(args []string) {
	fs := flag.NewFlagSet("install", flag.ExitOnError)
	src := fs.String("src", "/opt/uvm-insight/lib", "directory with libuvminsight.so and its CUPTI")
	dst := fs.String("dst", "/host/lib", "host directory to install into")
	_ = fs.Parse(args)

	if err := os.MkdirAll(*dst, 0o755); err != nil {
		log.Fatal(err)
	}
	entries, err := os.ReadDir(*src)
	if err != nil {
		log.Fatal(err)
	}
	for _, e := range entries {
		from := filepath.Join(*src, e.Name())
		to := filepath.Join(*dst, e.Name())
		changed, err := installOne(from, to)
		if err != nil {
			log.Fatalf("install %s: %v", e.Name(), err)
		}
		if changed {
			log.Printf("installed %s", to)
		}
	}
}

func installOne(from, to string) (bool, error) {
	info, err := os.Lstat(from)
	if err != nil {
		return false, err
	}
	tmp := filepath.Join(filepath.Dir(to), "."+filepath.Base(to)+".tmp")
	_ = os.Remove(tmp)

	if info.Mode()&os.ModeSymlink != 0 {
		target, err := os.Readlink(from)
		if err != nil {
			return false, err
		}
		if cur, err := os.Readlink(to); err == nil && cur == target {
			return false, nil
		}
		if err := os.Symlink(target, tmp); err != nil {
			return false, err
		}
		return true, os.Rename(tmp, to)
	}
	if !info.Mode().IsRegular() {
		return false, nil
	}
	want, err := fileSum(from)
	if err != nil {
		return false, err
	}
	if have, err := fileSum(to); err == nil && bytes.Equal(have, want) {
		return false, nil
	}
	if err := copyFile(from, tmp, info.Mode().Perm()); err != nil {
		_ = os.Remove(tmp)
		return false, err
	}
	return true, os.Rename(tmp, to)
}

func fileSum(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return nil, err
	}
	return h.Sum(nil), nil
}

func copyFile(from, to string, perm os.FileMode) error {
	in, err := os.Open(from)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(to, os.O_CREATE|os.O_EXCL|os.O_WRONLY, perm)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	if err := out.Sync(); err != nil {
		out.Close()
		return fmt.Errorf("sync: %w", err)
	}
	return out.Close()
}
