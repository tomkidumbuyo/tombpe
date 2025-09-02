from bpe.tokenizer import BasicTokenizer

if __name__ == "__main__":
    tokenizer = BasicTokenizer(vocab_size=32768)

    with open("test\short-text-file.txt", "r", encoding="utf-8") as f:
        training_data = f.read()
    tokenizer.train(training_data)
    tokenizer.save("vocab2.json")
    tokenized_string = tokenizer.encode(training_data)

    print("Tokenized string:", tokenized_string)

    detokenized_string = tokenizer.decode(tokenized_string)

    print("Detokenized string:", detokenized_string)
    
    
