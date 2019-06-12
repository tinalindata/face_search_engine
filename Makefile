CC=g++
DONKEY_HOME=donkey

EXTRA_CXXFLAGS=-I/usr/local/include -Idonkey/3rd/Simple-Web-Extra -Idonkey/3rd/Simple-Web-Server -Wno-enum-compare
EXTRA_HEADERS=
EXTRA_LDFLAGS=-static -L/usr/local/lib -L3rdparty/lib
EXTRA_LIBS= -lkgraph -lfmt -lssl -lcrypto -ldl -lz -lcares 
EXTRA_SOURCES=


include $(DONKEY_HOME)/src/Makefile.http.common

all: server
