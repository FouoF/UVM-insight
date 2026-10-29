package webui

import (
	"embed"
	"io/fs"
)

//go:embed all:dist
var embedded embed.FS

// Dist is the built web console. `make web` refreshes these files.
func Dist() fs.FS {
	sub, err := fs.Sub(embedded, "dist")
	if err != nil {
		return embedded
	}
	return sub
}
