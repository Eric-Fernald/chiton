//go:build windows

package model

import (
	"path/filepath"
	"strings"
	"sync"
	"syscall"
)

// canonicalDirs caches directories resolved to their on-disk form, since FileURI runs several times per file.
var canonicalDirs sync.Map

// canonicalPath returns abs with an upper-case drive letter and every existing component in its on-disk case,
// so c:\repo\x.go and C:\Repo\x.go name one file. A missing file keeps its given base name.
func canonicalPath(abs string) string {
	dir, base := filepath.Split(abs)
	if base == "" {
		return canonicalDir(filepath.Clean(abs))
	}
	if name, ok := onDiskName(abs); ok {
		base = name
	}
	return filepath.Join(canonicalDir(filepath.Clean(dir)), base)
}

// canonicalDir resolves dir with filepath.EvalSymlinks, which on Windows also normalises case, falling back to
// its parent for a directory that does not exist. Only successful resolutions are cached.
func canonicalDir(dir string) string {
	if v, ok := canonicalDirs.Load(dir); ok {
		return v.(string)
	}
	if c, err := filepath.EvalSymlinks(dir); err == nil {
		canonicalDirs.Store(dir, c)
		return c
	}
	parent := filepath.Dir(dir)
	if parent == dir {
		return upperDrive(dir)
	}
	return filepath.Join(canonicalDir(parent), filepath.Base(dir))
}

// onDiskName returns the stored name of path's last element, if it exists.
func onDiskName(path string) (string, bool) {
	p, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return "", false
	}
	var data syscall.Win32finddata
	h, err := syscall.FindFirstFile(p, &data)
	if err != nil {
		return "", false
	}
	_ = syscall.FindClose(h)
	return syscall.UTF16ToString(data.FileName[:]), true
}

func upperDrive(p string) string {
	if len(p) >= 2 && p[1] == ':' {
		return strings.ToUpper(p[:1]) + p[1:]
	}
	return p
}
