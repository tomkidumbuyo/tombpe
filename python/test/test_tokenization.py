from bpe.tokenizer import BasicTokenizer

if __name__ == "__main__":
    tokenizer = BasicTokenizer(vocab_size=32768)
    # training_data = "This is a very long training string. " * 2
    training_data = "abcabcabcabc"
    tokenizer.train(training_data)
    tokenizer.save("vocab.json")
    
