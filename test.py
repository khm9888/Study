t=int(input())
v=[float(input()) for f in range(t)]
v=[round(f*0.8,2) for f in v]
v=[print(f"${round(f,2):0.2f}") for f in v]

