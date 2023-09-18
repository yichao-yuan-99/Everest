import sys

output_started = False

for line in sys.stdin:
  line = line.strip()
  
  if line == "@@@ [output starts]":
    output_started = True
    continue
  
  if output_started:
    if line == "@@@ [output ends]":
      output_started = False
    else:
      print(line)