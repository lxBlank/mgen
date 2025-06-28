import json
import os
import difflib
f = open("model.bin.decode_beam10_jsonl.json")
objs = json.load(f)
iiiii = 0
for obj in objs:
    iiiii += 1
    name = obj["example"]["idx"]
    print(iiiii, name)
    #if iiiii < 12:
        #continue
    init = name + "_nonvul.c"
    gen = name + "_gen.c"
    tgt = name + "_vul.c"
    initcodeliststr = obj["example"]["init_code"]
    ftmp = open("tmp.py", "w")
    ftmp.write("tokenlist = " + initcodeliststr + "\nfor token in tokenlist:\n    if not token == '[DUMMY-REDUCE]':\n        print(token, end=' ')")
    ftmp.close()
    proc = os.popen("python tmp.py")
    initcode = proc.read()
    finit = open("./generated/" + init,"w")
    finit.write(initcode)
    finit.close()
    
    tgtcodeliststr = obj["example"]["tgt_code"]
    ftmp = open("tmp.py", "w")
    ftmp.write("tokenlist = " + tgtcodeliststr + "\nfor token in tokenlist:\n    if not token == '[DUMMY-REDUCE]':\n        print(token, end=' ')")
    ftmp.close()
    proc = os.popen("python tmp.py")
    tgtcode = proc.read()

    ftgt = open("./generated/" + tgt,"w")
    ftgt.write(tgtcode)
    ftgt.close()

    for iii in range(len(obj["hypotheses_logs"])):
        tokenliststr = obj["hypotheses_logs"][iii]["code"]
        tree = obj["hypotheses_logs"][iii]["tree"]
        flag = 0
        paraflag = 0
        startpoint = -1
        fgen = open("./generated/" + gen + str(iii),"w")
        for i in range(len(tree)):
            if tree[i:i+11] == "(Parameter " or tree[i:i+10] == "(Argument ":
                paraflag = 1
                parapar = 1
                continue
            if paraflag == 1 and tree[i] == '(' and flag == 0:
                parapar += 1
                continue
            if paraflag == 1 and tree[i] == ')' and flag == 0:
                parapar -= 1
                if parapar == 0 and (tree[i:i+12] == ") (Argument " or tree[i:i+13] == ") (Parameter "):
                    fgen.write(", ")
                    paraflag = 0
                    continue
            
            if flag == 0 and tree[i] == "'":
                flag = 1
                startpoint = i
                continue
            if flag == 1 and tree[i] == "'" and tree[i-1] != '\\':
                flag = 0
                token = tree[startpoint+1:i]
                token = token.replace('\\"', '"')
                token = token.replace("\\'", "'")
                token = token.replace('\\\\n', '\\n')
                token = token.replace('-SPACE-', ' ')
                if token != "[DUMMY-REDUCE]":
                    fgen.write(token+" ")
        fgen.close()
        os.system('clang-format -style=file ./generated/' + gen+str(iii) + '>' + './generated_formatted/' + gen+str(iii))
    
    os.system('clang-format -style=file ./generated/' + init + '>' + './generated_formatted/' + init)
    os.system('clang-format -style=file ./generated/' + tgt + '>' + './generated_formatted/' + tgt)
    
    for iii in range(len(obj["hypotheses_logs"])):
        fgen = open('./generated_formatted/' + gen+str(iii))
        finit = open('./generated_formatted/' + init)
        ftgt = open('./generated_formatted/' + tgt)

        fgenfix = open('./generated_fixed/' + gen+str(iii), "w")
        finitfix = open('./generated_fixed/' + init, "w")
        ftgtfix = open('./generated_fixed/' + tgt, 'w')

        genlines = fgen.readlines()
        initlines = finit.readlines()
        tgtlines = ftgt.readlines()

        diffs = ""
        initline_loc = -1
        genline_loc = -1
        for line in difflib.unified_diff(initlines, genlines):
            diffs = diffs + line
        diffx = diffs
        diffs = diffs.split("\n")
        tmpdiffs = []
        flag = 0
        minus = 0
        plus = 0
        mmm = 2
        startpoint = 0
        while mmm < len(diffs) and len(diffs[mmm])>0:
            if flag == 0 and diffs[mmm][0]== '-':
                startpoint = mmm
                flag = 1
                minus = 1
                mmm += 1
                continue
            if flag == 1 and diffs[mmm][0] == '-':
                minus += 1
                mmm += 1
                continue
            elif flag == 1 and diffs[mmm][0] == '+':
                flag = 2
                plus = 1
                mmm += 1
                continue
            if flag == 1 and (diffs[mmm][0] != '-' and diffs[mmm][0] != '+'):
                tmpdiffs.extend(diffs[startpoint:mmm])
                flag = 0
                plus = 0
                minus = 0
                #mmm += 1
                continue

            if flag == 2 and diffs[mmm][0] == '+':
                plus += 1
                mmm += 1
                continue
            if flag == 2 and diffs[mmm][0] != '+':
                if plus == minus:
                    for nnn in range(plus):
                        tmpdiffs.append(diffs[mmm-2*plus+nnn])
                        tmpdiffs.append(diffs[mmm-plus+nnn])
                    #mmm += 1
                    flag = 0
                    plus = 0
                    minus = 0
                    continue
                else:
                    tmpdiffs.extend(diffs[startpoint:mmm])
                    flag = 0
                    plus = 0
                    minus = 0
                    #mmm += 1
                    continue
            if flag == 0 and diffs[mmm][0] != '-':
                tmpdiffs.append(diffs[mmm])
                mmm += 1 
                continue
        
        #for diff in tmpdiffs:
            #print(diff)
        #import pdb
        #pdb.set_trace()
        diffs = tmpdiffs
        j = 0
        while j < len(diffs)-1:
            line = diffs[j]
            if line[:2] == '@@':
                try:
                    for i in range(len(line)):
                        if line[i] == '-':
                            initline_loc = int(line[i+1:].split(",")[0]) - 1
                        if line[i] == '+':
                            genline_loc = int(line[i+1:].split(",")[0]) - 1
                except:
                    break
            if line[0] == " ":
                initline_loc += 1
                genline_loc += 1
            if diffs[j][0] == "-" and diffs[j+1][0] == "+":
                for kkk,t in enumerate(genlines[genline_loc]):
                    if t!= ' ':
                        genline = genlines[genline_loc][kkk:]
                        break
                if "if ()" in genlines[genline_loc] and "if (" in initlines[initline_loc]:
                    genlines[genline_loc] = initlines[initline_loc]
                elif "for (; " in genlines[genline_loc] and "for (" in initlines[initline_loc]:
                    genlines[genline_loc] = initlines[initline_loc]
                elif abs(len(genlines[genline_loc]) - len(initlines[initline_loc])) <= 2:
                    genlines[genline_loc] = initlines[initline_loc]
                elif len(genline.split("=")[0].split(" ")) > 2:
                    genlines[genline_loc] = initlines[initline_loc]    
                elif '&' in genline or '|' in genline or '&' in initlines[initline_loc] or '|' in initlines[initline_loc] or '>>' in genline or '<<' in genline or '>>' in initlines[initline_loc] or '<<' in initlines[initline_loc]:
                    genlines[genline_loc] = initlines[initline_loc]
                initline_loc += 1
                genline_loc += 1
                j += 2
                continue
            elif diffs[j][0] == "-":
                initline_loc += 1
            elif diffs[j][0] == "+":
                genline_loc += 1
            j += 1
        for genline in genlines:
            fgenfix.write(genline)
        for initline in initlines:
            finitfix.write(initline)
        for tgtline in tgtlines:
            ftgtfix.write(tgtline)

        fgenfix.close()
        finitfix.close()
        ftgtfix.close()
        #import pdb
        #pdb.set_trace()



   
