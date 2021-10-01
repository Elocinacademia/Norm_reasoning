import csv
from pprint import pprint

file = open('output.csv')
reader = csv.reader(file)
header_row = next(reader)



for index, rows in enumerate(reader):
    j=1
    final = []
    if index > 2:
        pos = 0
        buffer = []
        prev = -1
        while pos < len(rows):
            if rows[pos] != prev and len(buffer) >= 6:
                final.append(buffer)
                buffer = [rows[pos]]
                prev = rows[pos]
            elif rows[pos] == prev:
                buffer.append(rows[pos])
            elif rows[pos] != prev:
                buffer = [rows[pos]]
                prev = rows[pos]
            pos += 1
        if len(buffer) >= 6:
            final.append(buffer)
        if len(final)>3:
            l =[]
            for item in final:
                l.append(len(item))
            maxvalue = max(l)
            # print('row number:': index, len(final), final)
            print('-------------------------------------')
            print('Row number:', index)
            print('Times of straightlining:', len(final))
            print('Maximum straightlining percentage:', maxvalue/70)
            print('Record:' , final)
            print('Source value:', rows)
            # print("row number:{0}, times of straightlining:{1}, Maximum straightlining percentage:{2} record: {3}".format(index, len(final), maxvalue/70, final))
    # if index == 83:
    #     print(len(rows))

 