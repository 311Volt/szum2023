import csv
import os
import itertools
import re
import random


def combination_string(gender_age_tuple):
    return "{} {}".format(gender_age_tuple[0], gender_age_tuple[1])


viable_combinations = [combination_string(p) for p in itertools.product(
    ["male", "female"],
    ["teens", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies"]
)]

all_sampleids = []
comb_sampleids = {comb_name: [] for comb_name in viable_combinations}
sampledata = dict()


def save_sampleids_info():
    with open("original_filenames.csv", "w") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["sampleid", "originalFilename"])
        writer.writeheader()
        for sampleid in all_sampleids:
            writer.writerow({
                "sampleid": sampleid,
                "originalFilename": sampledata[sampleid]["originalFilename"]
            })


def save_split(sampleids, output_filename):
    with open(output_filename, "w", newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["rowid", "sampleid", "age", "gender"])
        writer.writeheader()
        i = 0
        for sampleid in sampleids:
            writer.writerow({
                "rowid": str(i),
                "sampleid": sampleid,
                "age": sampledata[sampleid]["age"],
                "gender": sampledata[sampleid]["gender"]
            })
            i += 1


def create_split1():
    test_sampleids = random.sample(all_sampleids, int(len(all_sampleids) * 0.1))
    train_sampleids = list(set(all_sampleids) - set(test_sampleids))
    return train_sampleids, test_sampleids


def create_split2(train_size, test_size):
    test_sampleids = set(random.sample(all_sampleids, int(len(all_sampleids) * 0.1)))
    train_sampleids = set(all_sampleids) - set(test_sampleids)

    test_result = []
    train_result = []

    comb_sampleids_test = {
        comb_name: [x for x in idlist if x in test_sampleids]
        for (comb_name, idlist) in comb_sampleids.items()
    }
    comb_sampleids_train = {
        comb_name: [x for x in idlist if x in train_sampleids]
        for (comb_name, idlist) in comb_sampleids.items()
    }

    for i in range(train_size):
        combination = random.choice(viable_combinations)
        sampleid = random.choice(comb_sampleids_train[combination])
        train_result.append(sampleid)

    for i in range(test_size):
        combination = random.choice(viable_combinations)
        sampleid = random.choice(comb_sampleids_test[combination])
        test_result.append(sampleid)

    return train_result, test_result


def entry():
    print("fetching data...")
    with open("processed.csv", "r") as srccsv:
        reader = csv.DictReader(srccsv)
        for row in reader:
            sampleid = re.findall(r'\d+', row["newFilename"])[0]
            if os.path.exists("mel/{}.png".format(sampleid)):
                sampledata[sampleid] = {
                    "sampleid": sampleid,
                    "originalFilename": row["oldFilename"],
                    "age": row["age"],
                    "gender": row["gender"]
                }
                comb = combination_string((row["gender"], row["age"]))
                if comb in viable_combinations:
                    all_sampleids.append(sampleid)
                    comb_sampleids[comb].append(sampleid)

    print("creating split1...")
    split1 = create_split1()

    print("creating split2...")
    split2 = create_split2(100000, 10000)

    print("saving the results...")
    save_split(split1[0], "split1_train.csv")
    save_split(split1[1], "split1_test.csv")
    save_split(split2[0], "split2_train.csv")
    save_split(split2[1], "split2_test.csv")

if __name__ == '__main__':
    entry()
