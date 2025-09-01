from bpe.tokenizer import BasicTokenizer

if __name__ == "__main__":
    tokenizer = BasicTokenizer(vocab_size=32768)
    training_data = "abcabcabcabc"
    tokenizer.train(training_data)
    tokenized_string = tokenizer.encode(training_data)

    print("Tokenized string:", tokenized_string)

    detokenized_string = tokenizer.decode(tokenized_string)

    print("Detokenized string:", detokenized_string)
    tokenizer.save("vocab.json")
    
