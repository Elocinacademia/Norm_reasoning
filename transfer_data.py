
import csv


f = open('worksheet.csv')
reader = csv.reader(f)
header_row = next(reader)
useful_header = header_row[header_row.index('Q8_1'):header_row.index('Q21_7')+1]
# import pdb; pdb.set_trace()



reverse_code = {'1':'5',
'2':'4',
'3':'3',
'4':'2',
'5':'1'}

useful_header_index = []
reverse_code_index = []
reverse_question = ['Q9_3','Q11_1','Q12_4','Q14_2','Q17_3', 'Q20_1']
for item in useful_header:
    useful_header_index.append(header_row.index(item))
for item in reverse_question:
    reverse_code_index.append(header_row.index(item))




strongly_disagree = ['Strongly Disagree\n1','Strongly Disagree\n1\n','Strongly disagree\n1']
strongly_agree = ['Strongly Agree\n5\n','Strongly Agree\n5','Strongly Agree5']
neutral_iu = ['Neutral\n4']
neutral_normal = ['Neutral3']
to_five = ['Strongly Agree\n5\n','Strongly Agree\n5','Strongly Agree5', 'Describes very well\n5\n']
to_one = ['Strongly Disagree\n1','Strongly Disagree\n1\n','Strongly disagree\n1','Describes very poorly\n1\n', 'Not at all\n1\n']




m = []
m.append(header_row)
for row in reader:
    for index, i in enumerate(row):
        # if i in strongly_disagree:
        #     row[index] = '1'
        # if i in strongly_agree:
        #     row[index] = '5'
        if i in to_five:
            row[index] = '5'
        if i in neutral_iu:
            row[index] = '4'
        if i in to_one:
            row[index] = '1'
        # if i == 'Describes very poorly\n1\n':
        #     row[index] = '1'
        # if i == 'Describes very well\n5\n':
        #     row[index] = '5'
        if i == 'Strongly agree\n7':
            row[index] = '7'
        # if i == 'Not at all\n1\n':
        #     row[index] = '1'
        if i in neutral_normal:
            row[index] = '3'
    m.append(row)
    # m.append(row[useful_header])
# import pdb; pdb.set_trace()

for row in m:
    for index, i in enumerate(row):
        if index in reverse_code_index and len(i) <= 1:
            row[index] = reverse_code[i]
    # print(isinstance(row,list))
# import pdb; pdb.set_trace()

with open("output.csv", "w", newline = '', encoding = 'utf8') as file:
    writer = csv.writer(file, delimiter = ',')
    # for rows in m[1:]:
    for rows in m:
        print(rows)
        # import pdb; pdb.set_trace()
        writer.writerow(rows)







