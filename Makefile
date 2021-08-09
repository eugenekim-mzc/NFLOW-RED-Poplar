CXX = c++ 
CXXFLAGS = -w -g -std=c++11 
LIBS = -lpoplar 

OBJS = main.o initialize.o runSimulation.o inverseMRT.o decomposition.o
SRCS = main.cpp initialize.cpp runSimulation.cpp inverseMRT.cpp decomposition.cpp
TARGET = test.x

all : $(TARGET)
$(TARGET) : $(OBJS)
	$(CXX) -o $@ $(OBJS) ${LIBS} 

clean:
	rm -rf $(OBJS) $(TARGET) core 
