all:
	python eri_OS_generator.py > out.c
	gcc -fPIC -O2 -g -shared -o liberi_OS.so out.c
	rm out.c
