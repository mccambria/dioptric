# -*- coding: utf-8 -*-


for ind in range(10):
    try:
        if ind - 5 < 0:
            raise Exception("Crashing")
        break
    except Exception as exc:
        print(exc)

print("Done")
