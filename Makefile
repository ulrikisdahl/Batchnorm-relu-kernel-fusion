.PHONY: all runMain
all: runMain

runMain: 
	@cd build && cmake ..
	@cd build && cmake --build .