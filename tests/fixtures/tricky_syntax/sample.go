package server

import (
	"context"
	"fmt"
	"net/http"
)

type WorkerPool[T any, R any] interface {
	Submit(task T) R
	Close() error
}

type Server struct {
	addr string
	pool WorkerPool[string, int]
}

func NewServer(addr string) *Server {
	initMetrics()
	return &Server{addr: addr}
}

func (s *Server) HandleRequest(
	ctx context.Context,
	req *http.Request,
) (*http.Response, error) {
	s.logAccess(req.URL.Path)
	res := processRequest(ctx, req)
	return res, nil
}

func (s *Server) logAccess(path string) {
	fmt.Println("Access:", path)
}
