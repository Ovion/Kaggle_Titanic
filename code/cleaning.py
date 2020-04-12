from cleaning_functions import read_data, create_court, fillna_age, last_clean_dummy


if __name__ == '__main__':
    train, test = read_data()

    print('Cleaning data...')
    train = create_court(train)
    test = create_court(test)

    train = fillna_age(train)
    test = fillna_age(test)

    train = last_clean_dummy(train)
    test = last_clean_dummy(test)

    print("Saving data at 'inputs'...")
    train.to_csv('inputs/train_clean.csv', index=False)
    test.to_csv('inputs/test_clean.csv', index=False)
