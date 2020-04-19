f = open("anomalousTrafficTest.txt", "r")

raw_text = ""
data_list = []
before_line = "initial"

while (True):
    line = f.readline()
    if line:
        if line == '\n' and "Connection:" in before_line:
            data_list.append({1:"{0}".format(raw_text)})
            raw_text = ""
            before_line = line
        elif ("GET" not in line) and ("POST" not in line) and line != '\n' and before_line == '\n':
            raw_text += line
            data_list.append({1:"{0}".format(raw_text)})
            raw_text = ""
            before_line = line
        else:
            if line == '\n' and before_line == '\n':
                pass
            elif line == '\n' and "Content-Length:" not in before_line:
                pass
            else:
                raw_text += line
                before_line = line
    else:
        break

print(data_list)
