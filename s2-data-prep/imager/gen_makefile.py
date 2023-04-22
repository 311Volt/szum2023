
max_range = 150000
step = 1000

chunknames = ""

for i in range(0, max_range, step):
    range_start = i
    range_end = i + step
    chunkname = "chunk_{:04d}".format(int(i/step))
    print("{}:\n\tpython imager.py.lnk {} {}\n\n".format(chunkname, range_start, range_end))
    chunknames += chunkname + " "

print("all: " + chunknames)