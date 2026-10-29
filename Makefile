IMAGE ?= dynamia-ai-registry.cn-hangzhou.cr.aliyuncs.com/artifacts/dynamia-ai/uvm-insight
TAG ?= dev

.PHONY: all web agent test demo collector examples image

all: agent

web:
	cd web && npm install && npm run build
	rm -rf agent/pkg/webui/dist
	mkdir -p agent/pkg/webui/dist
	cp -R web/dist/. agent/pkg/webui/dist/

agent: web
	cd agent && CGO_ENABLED=0 go build -o ../bin/uvm-insight-agent ./cmd/uvm-insight-agent

test:
	mkdir -p bin
	c++ -std=c++17 -I collector/src collector/src/encode.cpp collector/src/encode_test.cpp -o bin/encode_test
	./bin/encode_test proto/testdata/sample.frames
	cd agent && go test ./...

demo: agent
	./bin/uvm-insight-agent --mock --listen 127.0.0.1:9400

collector:
	cmake -S collector -B collector/build
	cmake --build collector/build -j

examples:
	cmake -S examples -B examples/build
	cmake --build examples/build -j

image:
	docker buildx build --platform linux/amd64 -f deploy/Dockerfile -t $(IMAGE):$(TAG) --load .
