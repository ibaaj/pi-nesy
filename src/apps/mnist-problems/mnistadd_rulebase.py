import sys

def generate_rule_base_mnist_add_problem(k: int, upper_bound: int = 10):
    if k < 1:
        print("k must be at least 1.")
        sys.exit(-1)
    
    attributes_set = []
    domain_attributes_set = {}
    

    for i in range(1,k+1):
        attributes_set.append("a_" + str(i))
        attributes_set.append("a_" + str(k+i))
        domain_attributes_set["a_" + str(i)] = [x for x in range(upper_bound)]
        domain_attributes_set["a_" + str(k+i)] = [x for x in range(upper_bound)]
    
    for i in range(k, -1, -1):
        if i == k:
            attributes_set.append("c_" +str(i))
            domain_attributes_set["c_" + str(i)] =  [(x, y) for x in range(upper_bound) for y in range(upper_bound)]
        elif i != 0:
            attributes_set.append("c_" +str(i))
            domain_attributes_set["c_" + str(i)] =  [(x, y, r) for x in range(upper_bound) for y in range(upper_bound) for r in [0,1]]

        if i >= 0:
            attributes_set.append("w_" +str(i))
            domain_attributes_set["w_" + str(i)] = [0,1]

        attributes_set.append("y_" + str(i))
        if i == 0:
             domain_attributes_set["y_" + str(i)] = [0,1]
        else:
            domain_attributes_set["y_" + str(i)] = [x for x in range(upper_bound)]


    print("attributes: " + str(attributes_set))

    """ c_tuples """
    rule_sets = {}
    rule_sets["c_tuples"] = {}
    for j in range(k,0,-1):
        rule_sets["c_tuples"]["c_" + str(j)] = []
        if j == k:
            for i in range(0,upper_bound):
                qset1 = [(i, y) for y in range(upper_bound)]
                rule = {"premise": {"a_" + str(j): [i], "a_" + str(k+j): []},
                        "conclusion": {
                            "c_" + str(j): qset1,
                            "excluded_values_of_c_" + str(j): [x for x in domain_attributes_set["c_" + str(j)] if x not in qset1]
                        }}
                rule_sets["c_tuples"]["c_" + str(j)].append(rule)

                qset2 = [(x, i) for x in range(upper_bound)]
                rule = {"premise": {"a_" + str(j): [], "a_" + str(k+j): [i]},
                        "conclusion": {
                            "c_" + str(j): qset2,
                            "excluded_values_of_c_" + str(j): [x for x in domain_attributes_set["c_" + str(j)] if x not in qset2]
                        }}
                rule_sets["c_tuples"]["c_" + str(j)].append(rule)
            print("computing " + "c_" + str(j) + " by: (" + "a_" + str(j) + " + " + "a_" + str(k+j) + " >= 10)")
        else:
            for i in range(0,upper_bound):
                qset1 = [(i, y, r) for y in range(upper_bound) for r in [0,1]]
                rule = {"premise": {"a_" + str(j): [i], "a_" + str(k+j): [], "w_" + str(j+1): []},
                        "conclusion": {
                            "c_" + str(j): qset1,
                            "excluded_values_of_c_" + str(j): [x for x in domain_attributes_set["c_" + str(j)] if x not in qset1]
                        }}
                rule_sets["c_tuples"]["c_" + str(j)].append(rule)

                qset2 = [(x, i, r) for x in range(upper_bound) for r in [0,1]]
                rule = {"premise": {"a_" + str(j): [], "a_" + str(k+j): [i]},
                        "conclusion": {
                            "c_" + str(j): qset2,
                            "excluded_values_of_c_" + str(j): [x for x in domain_attributes_set["c_" + str(j)] if x not in qset2]
                        }}
                rule_sets["c_tuples"]["c_" + str(j)].append(rule)
            
            qset1 = [(x, y, 0) for x in range(upper_bound) for y in range(upper_bound)]
            rule = {"premise": {"a_" + str(j): [], "a_" + str(k+j): [], "w_" + str(j+1): [0]},
                    "conclusion": {
                        "c_" + str(j): qset1,
                        "excluded_values_of_c_" + str(j): [x for x in domain_attributes_set["c_" + str(j)] if x not in qset1]
                    }}
            rule_sets["c_tuples"]["c_" + str(j)].append(rule)

            print("computing " + "c_" + str(j) + " by: (" + "a_" + str(j) + " + " + "a_" + str(k+j) + " + w_" + str(j+1) + ") >= 10)")
        
        
    
    """ carry """ 
    rule_sets["carry"] = {}
    for j in range(k,0,-1):
        rule_sets["carry"]["w_" + str(j)] = []
        if j == k:
            rule = {
                    "premise": { 
                        "c_" + str(j): [(x, y) for x in range(upper_bound) for y in range(upper_bound) if x+y >= 10],
                        "excluded_values_of_c_" + str(j): [(x, y) for x in range(upper_bound) for y in range(upper_bound) if x+y < 10]
                        },
                    "conclusion": {
                        "w_" + str(j): [1],
                        "excluded_values_of_w_" + str(j): [0]
                    }}
            rule_sets["carry"]["w_" + str(j)].append(rule)
        else:
            rule = {
                    "premise": { 
                        "c_" + str(j): [(x, y, r) for x in range(upper_bound) for y in range(upper_bound) for r in [0,1] if x+y+r >= 10],
                        "excluded_values_of_c_" + str(j): [(x, y, r) for x in range(upper_bound) for y in range(upper_bound) for r in [0,1] if x+y+r < 10]
                        },
                    "conclusion": {
                        "w_" + str(j): [1],
                        "excluded_values_of_w_" + str(j): [0]
                    }}
            rule_sets["carry"]["w_" + str(j)].append(rule)
        print("computing " + "w_" + str(j) + " by: c_" + str(j))
       

    """ y_i """ 
    rule_sets["y_outputs"] = {}

    for j in range(k,-1,-1):
        rule_sets["y_outputs"]["y_" + str(j)] = []
        if j == k:
            for i in range(upper_bound):
                pset = [(u, v) for u in range(upper_bound) for v in range(upper_bound) if (u + v) % 10 == i]
                rule = {"premise": {"c_" + str(j): pset,
                                "excluded_values_of_c_" + str(j): [(u, v) for u in range(upper_bound) for v in range(upper_bound) if (u, v) not in pset]
                                },
                    "conclusion": {"y_" + str(j): [i],
                                "excluded_values_of_y_" + str(j): [x for x in domain_attributes_set["y_" + str(j)] if x != i]}}
                rule_sets["y_outputs"]["y_" + str(j)].append(rule)
        elif j == 0:
            rule = {"premise": {
                                "w_1": [0],
                                "excluded_values_of_w_1": [1]
                                },
                    "conclusion": {"y_0": [0],
                                "excluded_values_of_y_0": [1]}}
            rule_sets["y_outputs"]["y_" + str(j)].append(rule)
        else:
            for i in range(upper_bound):
                pset = [(u, v, r) for u in range(upper_bound) for v in range(upper_bound) for r in [0,1] if (u + v + r) % 10 == i]
                rule = {"premise": {"c_" + str(j): pset,
                                "excluded_values_of_c_" + str(j): [(u, v, r) for u in range(upper_bound) for v in range(upper_bound) for r in [0,1] if (u, v, r) not in pset]
                                },
                    "conclusion": {"y_" + str(j): [i],
                                "excluded_values_of_y_" + str(j): [x for x in domain_attributes_set["y_" + str(j)] if x != i]}}
                rule_sets["y_outputs"]["y_" + str(j)].append(rule)
        if j != 0:
            print("computing y_"+str(j) +" by: c_" + str(j))
        else:
            print("computing y_"+str(j) +" by: w_" + str(j+1))

    return attributes_set, domain_attributes_set, rule_sets

