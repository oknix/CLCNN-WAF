import csv

def make_csv(filename, flag):
    with open(filename, "r") as f:
        raw_text = ""
        data_list = []
        before_line = "initial"

        while True:
            line = f.readline()
            if line:
                if line == '\n' and "Connection:" in before_line:
                    data_list.append(["{0}".format(flag),"{0}".format(raw_text)])
                    raw_text = ""
                    before_line = line
                elif ("GET" not in line) and ("POST" not in line) and ("PUT" not in line) and line != '\n' and before_line == '\n':
                    raw_text += line
                    data_list.append(["{0}".format(flag),"{0}".format(raw_text)])
                    raw_text = ""
                    before_line = line
                elif line == '\n' and before_line == '\n':
                    pass
                elif line == '\n' and "Content-Length:" not in before_line:
                    pass
                else:
                    raw_text += line
                    before_line = line
            else:
                break

    with open(filename.replace("txt", "csv"), "w") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerows(data_list)

if __name__ == "__main__":
    make_csv("anomalousTrafficTest.txt", "1")
    make_csv("normalTrafficTraining.txt", "0")
