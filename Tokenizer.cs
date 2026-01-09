using System.Collections.Generic;
using System.Linq;
using System.Text;

public class Tokenizer
{
    // Словари
    private Dictionary<string, int> russianVocab = new Dictionary<string, int>();
    private Dictionary<string, int> englishVocab = new Dictionary<string, int>();
    private Dictionary<int, string> russianReverse = new Dictionary<int, string>();
    private Dictionary<int, string> englishReverse = new Dictionary<int, string>();
    
    // Специальные токены
    public const int PAD_TOKEN = 0;
    public const int SOS_TOKEN = 1;  // Start of Sequence
    public const int EOS_TOKEN = 2;  // End of Sequence
    public const int UNK_TOKEN = 3;  // Unknown word
    
    private int russianIndex = 4;
    private int englishIndex = 4;
    
    public Tokenizer(List<string> russianSentences, List<string> englishSentences)
    {
        // Инициализация специальных токенов
        russianVocab["<PAD>"] = PAD_TOKEN;
        russianVocab["<SOS>"] = SOS_TOKEN;
        russianVocab["<EOS>"] = EOS_TOKEN;
        russianVocab["<UNK>"] = UNK_TOKEN;
        
        englishVocab["<PAD>"] = PAD_TOKEN;
        englishVocab["<SOS>"] = SOS_TOKEN;
        englishVocab["<EOS>"] = EOS_TOKEN;
        englishVocab["<UNK>"] = UNK_TOKEN;
        
        // Построение словарей
        BuildVocabulary(russianSentences, russianVocab, ref russianIndex);
        BuildVocabulary(englishSentences, englishVocab, ref englishIndex);
        
        // Создание обратных словарей
        CreateReverseDictionaries();
    }
    
    private void BuildVocabulary(List<string> sentences, Dictionary<string, int> vocab, ref int index)
    {
        foreach (var sentence in sentences)
        {
            var words = CleanAndSplit(sentence);
            foreach (var word in words)
            {
                if (!vocab.ContainsKey(word))
                {
                    vocab[word] = index++;
                }
            }
        }
    }
    
    private List<string> CleanAndSplit(string text)
    {
        var cleaned = new StringBuilder();
        foreach (char c in text.ToLower())
        {
            if (char.IsLetterOrDigit(c) || c == ' ' || c == '\'')
            {
                cleaned.Append(c);
            }
            else if (char.IsPunctuation(c))
            {
                cleaned.Append(' ');
            }
        }
        
        return cleaned.ToString()
            .Split(' ', System.StringSplitOptions.RemoveEmptyEntries)
            .ToList();
    }
    
    // Токенизация русского предложения
    public List<int> TokenizeRussian(string sentence)
    {
        var tokens = new List<int> { SOS_TOKEN };
        var words = CleanAndSplit(sentence);
        
        foreach (var word in words)
        {
            if (russianVocab.TryGetValue(word, out int token))
            {
                tokens.Add(token);
            }
            else
            {
                tokens.Add(UNK_TOKEN);
            }
        }
        tokens.Add(EOS_TOKEN);
        
        return tokens;
    }
    
    // Токенизация английского предложения
    public List<int> TokenizeEnglish(string sentence)
    {
        var tokens = new List<int> { SOS_TOKEN };
        var words = CleanAndSplit(sentence);
        
        foreach (var word in words)
        {
            if (englishVocab.TryGetValue(word, out int token))
            {
                tokens.Add(token);
            }
            else
            {
                tokens.Add(UNK_TOKEN);
            }
        }
        tokens.Add(EOS_TOKEN);
        
        return tokens;
    }
    
    // Детокенизация
    public string DetokenizeRussian(List<int> tokens)
    {
        var words = new List<string>();
        foreach (var token in tokens)
        {
            if (token == EOS_TOKEN) break;
            if (token == SOS_TOKEN || token == PAD_TOKEN) continue;
            
            if (russianReverse.TryGetValue(token, out string word))
            {
                words.Add(word);
            }
        }
        return string.Join(" ", words);
    }
    
    public string DetokenizeEnglish(List<int> tokens)
    {
        var words = new List<string>();
        foreach (var token in tokens)
        {
            if (token == EOS_TOKEN) break;
            if (token == SOS_TOKEN || token == PAD_TOKEN) continue;
            
            if (englishReverse.TryGetValue(token, out string word))
            {
                words.Add(word);
            }
        }
        return string.Join(" ", words);
    }
    
    private void CreateReverseDictionaries()
    {
        foreach (var kvp in russianVocab)
        {
            russianReverse[kvp.Value] = kvp.Key;
        }
        
        foreach (var kvp in englishVocab)
        {
            englishReverse[kvp.Value] = kvp.Key;
        }
    }
    
    public int GetRussianVocabSize() => russianVocab.Count;
    public int GetEnglishVocabSize() => englishVocab.Count;
}