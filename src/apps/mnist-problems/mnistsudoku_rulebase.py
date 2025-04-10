import sys
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')



def generate_rule_base_sudoku(F,SIZE):
    AttributesSet = ["c"]
    DomainAttributesSet = {}

    for i in range(0,SIZE):
        for j in range(0,SIZE):
            AttributesSet.append("a_" + str(i) + "_" + str(j))
            DomainAttributesSet["a_" + str(i) + "_" + str(j)] = [x for x in range(0, SIZE)]
            logging.info(DomainAttributesSet["a_" + str(i) + "_" + str(j)])

    for f in F:
        AttributesSet.append("b_" + str(f))
        DomainAttributesSet["b_" + str(f)] = [(x, y) for x in range(0, SIZE) for y in range(0, SIZE)]


    DomainAttributesSet["c"] = [0, 1]

    FirstSetsRules = {}
    SecondSetRules = []
    for f in F:
        i,j,ip,jp = f
        SetOfRules = []
        for k in range(0, SIZE):
            R1 = {}
            R2 = {}
            R1["premise"] = {}
            R2["premise"] = {}
            R1["conclusion"] = {}
            R2["conclusion"] = {}
            R1["premise"]["a_" + str(i) + "_" + str(j)] = [k]
            R2["premise"]["a_" + str(ip) + "_" + str(jp)] = [k]
            R1["conclusion"]["b_" + str(f)] = [(x,y) for x in range(0, SIZE) for y in range(0, SIZE) if x == k]
            R2["conclusion"]["b_" + str(f)] = [(x,y) for x in range(0, SIZE) for y in range(0, SIZE) if y == k]

            R1["conclusion"]["excluded_values_of_b_" + str(f)] = [x for x in DomainAttributesSet["b_" + str(f)] if
                                                          x not in R1["conclusion"]["b_" + str(f)]]
            R2["conclusion"]["excluded_values_of_b_" + str(f)] = [x for x in DomainAttributesSet["b_" + str(f)] if
                                                                 x not in R2["conclusion"]["b_" + str(f)]]
            SetOfRules.append(R1)
            SetOfRules.append(R2)

        FirstSetsRules[f] = SetOfRules

    Rule = {}
    Rule["premise"] = {}
    Rule["conclusion"] = {}
    for f in F:
        Rule["premise"]["b_" + str(f)] = [(x,y) for x in range(0, SIZE) for y in range(0, SIZE) if x != y]
        Rule["premise"]["excluded_values_of_b_" + str(f)] = [x for x in DomainAttributesSet["b_" + str(f)] if
                                                   x not in Rule["premise"]["b_" + str(f)]]

    Rule["conclusion"]["c"] = [1]
    Rule["conclusion"]["excluded_values_of_c"] = [0]

    SecondSetRules.append(Rule)

    return (AttributesSet, DomainAttributesSet, FirstSetsRules, SecondSetRules)