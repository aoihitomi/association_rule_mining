import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 数据加载与预处理
def dataset_load_and_preprocess():
    # 读入数据
    dataset = pd.read_csv(".\\winemag-data_first150k.csv")
    # 去除包含空值的行
    dataset = dataset.dropna()
    # 去除不关心的列
    dataset = dataset[['country', 'points', 'price', 'province', 'region_1', 'region_2', 'variety', 'winery']]

    # 将数据集转换为方便数据挖掘的形式 -> 二维表转换为一维列表
    row, col = dataset.shape

    data = []
    for i in range(row):
        datarow = []
        for j in range(col):
            datarow.append((dataset.columns[j], dataset.iloc[i, j]))
        data.append(datarow)
    
    return data

# 创建1-频繁项集的候选集
def create_C1(data_set):
    C1 = set()
    for t in data_set:
        for item in t:
            item_set = frozenset([item])
            C1.add(item_set)
    return C1

# 判断候选集是否满足apriori规则
def is_apriori(Ck_item, Lksub1):
    for item in Ck_item:
        sub_Ck = Ck_item - frozenset([item])
        if sub_Ck not in Lksub1:
            return False
    return True

# 创建k-频繁项集的候选集
def create_Ck(Lksub1, k):
    Ck = set()
    len_Lksub1 = len(Lksub1)
    list_Lksub1 = list(Lksub1)
    for i in range(len_Lksub1):
        for j in range(1, len_Lksub1):
            l1 = list(list_Lksub1[i])
            l2 = list(list_Lksub1[j])
            l1.sort()
            l2.sort()
            if l1[0:k-2] == l2[0:k-2]:
                Ck_item = list_Lksub1[i] | list_Lksub1[j]
                if is_apriori(Ck_item, Lksub1):
                    Ck.add(Ck_item)
    return Ck

# 由k-频繁项集候选集生成实际的频繁项集
def generate_Lk_by_Ck(data_set, Ck, min_support, support_data):
    Lk = set()
    item_count = {}
    for t in data_set:
        for item in Ck:
            if item.issubset(t):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1
    t_num = float(len(data_set))
    for item in item_count:
        if (item_count[item] / t_num) >= min_support:
            Lk.add(item)
            support_data[item] = item_count[item] / t_num
    return Lk

# 生成从1-频繁项集到k-频繁项集的全集
def generate_L(data_set, k, min_support):
    support_data = {}
    C1 = create_C1(data_set)
    L1 = generate_Lk_by_Ck(data_set, C1, min_support, support_data)
    Lksub1 = L1.copy()
    L = []  
    L.append(Lksub1)
    for i in range(2, k+1):
        Ci = create_Ck(Lksub1, i)
        Li = generate_Lk_by_Ck(data_set, Ci, min_support, support_data)
        Lksub1 = Li.copy()
        L.append(Lksub1)
    return L, support_data

# 由频繁项集挖掘关联规则
def generate_rules(L, support_data, min_conf):
    rules_list = []
    sub_set_list = []
    for i in range(0, len(L)):
        for freq_set in L[i]:
            for sub_set in sub_set_list:
                if sub_set.issubset(freq_set):
                    conf = support_data[freq_set] / support_data[freq_set - sub_set]
                    rule = (freq_set - sub_set, sub_set, conf)
                    if conf >= min_conf and rule not in rules_list:
                        rules_list.append(rule)
            sub_set_list.append(freq_set)
    return rules_list

# 计算关联规则的lift指标和余弦相似度
def calculate_lift_and_cosine(a, b, support_data):
    support_ab = 0
    if (a | b) in support_data.keys():
        support_ab = support_data[a | b]
    
    lift = support_ab / support_data[a] / support_data[b]
    cosine = support_ab / (support_data[a] * support_data[b]) ** 0.5
    return lift, cosine


def plot(rules_list, support_data):
    row = []
    col = []
    length = len(rules_list)
    for item in rules_list:
        row.append(item[0])
        col.append(item[1])
    
    sim_lift = []
    sim_cosine = []
    for i in range(length):
        datarow_lift = []
        datarow_cosine = []
        for j in range(length):
            lift, cosine = calculate_lift_and_cosine(row[i], col[j], support_data)
            datarow_lift.append(lift)
            datarow_cosine.append(cosine)
        sim_lift.append(datarow_lift)
        sim_cosine.append(datarow_cosine)
    
    plt.title('Lift')
    sns.heatmap(sim_lift, cmap='Reds')
    plt.show()
    
    plt.title('Cosine')
    sns.heatmap(sim_lift, cmap='Reds')
    plt.show()


if __name__ == "__main__":
    data_set = dataset_load_and_preprocess()
    L, support_data = generate_L(data_set, k=3, min_support=0.1)
    rules_list = generate_rules(L, support_data, min_conf=0.7)
    
    cnt = 1
    for Lk in L:
        print("="*50)
        print("frequent " + str(cnt) + "-itemsets\t\tsupport")
        cnt += 1
        print("="*50)
        for freq_set in Lk:
            for freq_item in freq_set:
                print(freq_item, end=' ')
            print(support_data[freq_set])
    print()
    print("Rules")
    for rules in rules_list:
        print("{", end='')
        for rule_item in rules[0]:
            print(rule_item, end=' ')
        print("} => {", end='')
        for rule_item in rules[1]:
            print(rule_item, end=' ')
        print("}} conf: {}".format(rules[2]))
        
    for rules in rules_list:
        print("{", end='')
        for rule_item in rules[0]:
            print(rule_item, end=' ')
        print("} => {", end='')
        for rule_item in rules[1]:
            print(rule_item, end=' ')
        lift, cosine = calculate_lift_and_cosine(rules[0], rules[1], support_data)
        print("}} Lift: {} Cosine: {}".format(lift, cosine))
    
    plot(rules_list, support_data)