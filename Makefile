KERNEL_DIR := ./kernels
OBJ_DIR := ./objects
TARGET := sw_emu
INCLUDES := -I/tools/Xilinx/Vivado/2020.1/include/ -I./xf_solver/L2/include/ -I./
LIBS := -L/opt/xilinx/xrt/lib/ -L/usr/local/lib -L/usr/lib -lOpenCL -lcnpy -lz -lpthread -lrt -lstdc++ -std=c++14 -w -O0 -g

SOURCES := $(wildcard ./kernels/*.cpp)
OBJECTS := $(patsubst %.cpp,%.xo,$(addprefix objects/,$(notdir $(SOURCES))))

$(info $$var is [${OBJECTS}])

kernels.xclbin: $(OBJECTS) 
	v++ -t $(TARGET) --config design.cfg -l -g -o $@ $^

$(OBJ_DIR)/%.xo: $(KERNEL_DIR)/%.cpp
	v++ -t $(TARGET) --config design.cfg -c -k $(basename $(notdir $<)) $(INCLUDES) -o $@ $<

host: host.cpp
	g++ -o host host.cpp -I/opt/xilinx/xrt/include/ -I/tools/Xilinx/Vivado/2020.1/include/ -I./xcl2/ -L/opt/xilinx/xrt/lib/ -L/usr/local/lib -L/usr/lib -lOpenCL -lcnpy -lz -lpthread -llapack -lrt -lstdc++ -std=c++14 -w -O0 -g

clean:
	rm -f *.log
	rm -f *_summary
	rm -f *.info
	rm -rf ./_x
	rm -rf ./.run
	rm -rf ./.Xil
	rm ./objects/*compile_summary
