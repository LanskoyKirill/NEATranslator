using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using Unity.VisualScripting;

public class AI : MonoBehaviour
{
    public GameObject spawnerOfNN;
    public bool go = true;
    public int cIterator = 0;
    public int addition = 0;

    // NEAT параметры
    public float speed = 0.1f;
    public int recursionAddLink = 0;
    public float howIsGood = 0;
    public float textHowIsGood = 0;
    private float accumulatedFitness = 0f;
    private int processedInBatch = 0;
    public int maxIterations = 1000;
    private int iterations = 0;

    // Структура сети
    public List<float> neurones = new List<float>();
    public List<int> inpInnov = new List<int>();
    public List<int> outInnov = new List<int>();
    public List<float> weights = new List<float>();
    public List<bool> actConnect = new List<bool>();
    public List<bool> RNNs = new List<bool>();
    public List<float> RNNneurones = new List<float>();
    public List<int> order = new List<int>();
    public List<Dictionary<int, float>> adjList = new List<Dictionary<int, float>>();
    public List<int> innovations = new List<int>();

    public int testing = 0;

    // Данные
    public string request;
    public string answer;
    public string answerNN;
    public bool ifNew = false;
    
    // Токенизированные данные
    public List<int> requestTokens = new List<int>();
    public List<int> answerTokens = new List<int>();
    public List<int> outputTokens = new List<int>();

    public int prevNumber = 0;
    public int thisNumber = 0;
    private int outConnections = 16;
    private int initalNeurones;
    public int desiredNeurones;

    // Состояние
    private bool isGenerating = false;
    private int currentTokenIndex = 0;
    private int nextInputToken = 1; // SOS token
    
    // Токенизатор
    [NonSerialized] public Tokenizer tokenizer;
    [NonSerialized] public int russianVocabSize;
    [NonSerialized] public int englishVocabSize;

    private HashSet<int> availableRussianTokens = new HashSet<int>();
    private HashSet<int> availableEnglishTokens = new HashSet<int>();
    private List<int> availableInputNeurons = new List<int>();
    private List<int> availableOutputNeurons = new List<int>();

    public void Start()
    {
        go = true;
        isGenerating = false;
        howIsGood = 0;
        textHowIsGood = 0;
        cIterator = 0;
        currentTokenIndex = 0;
        nextInputToken = 1; // SOS
        answerNN = "";
        outputTokens.Clear();
        
        // Инициализация нейронов
        initalNeurones = 1 + russianVocabSize + englishVocabSize;
        
        if (neurones.Count < initalNeurones)
        {
            neurones.Clear();
            RNNneurones.Clear();
            neurones.Add(1f);
            RNNneurones.Add(1f);
            for (int i = 0; i < initalNeurones - 1; i++)
            {
                neurones.Add(0f);
                RNNneurones.Add(0f);
            }
        }
        else
        {
            neurones[0] = 1f;
            RNNneurones[0] = 1f;
            for (int i = 1; i < neurones.Count; i++)
            {
                neurones[i] = 0f;
                RNNneurones[i] = 0f;
            }
        }
        RNNneurones = new List<float>(neurones);
        
        // Удаление невалидных связей
        for (int i = 0; i < inpInnov.Count; i++)
        {
            if (inpInnov[i] >= neurones.Count || outInnov[i] >= neurones.Count)
            {
                inpInnov.RemoveAt(i);
                outInnov.RemoveAt(i);
                weights.RemoveAt(i);
                actConnect.RemoveAt(i);
                RNNs.RemoveAt(i);
                innovations.RemoveAt(i);
                i--;
            }
        }

        UpdateAvailableNeurons();
        
        makeOrder();
        Adder();
        
        // Загрузка материала
        Material AIMaterial = Resources.Load(spawnerOfNN.name, typeof(Material)) as Material;
        if (AIMaterial != null)
        {
            gameObject.GetComponent<Renderer>().material = AIMaterial;
        }
        
        if (order.Count > 3 && order[0] == 1 && order[1] == 0 && order[2] == 0)
        {
            ++testing;
        }
        
        // Токенизация если токенизатор есть
        if (tokenizer != null && !string.IsNullOrEmpty(request) && !string.IsNullOrEmpty(answer))
        {
            requestTokens = tokenizer.TokenizeRussian(request);
            answerTokens = tokenizer.TokenizeEnglish(answer);
        }

        //Debug.Log($"AI Start. Tokenizer: {tokenizer != null}, Request: {request}, Answer: {answer}");
    }
    
    void Update()
    {
        if (!inpInnov.Any())
        {
            //++spawnerOfNN.GetComponent<GameManager>().readyToCheck;
            Destroy(gameObject);
            return;
        }
        
        if (go == true)
        {
            neurones[0] = 1;
            
            // Если нет токенизатора - используем старую логику
            if (tokenizer == null || requestTokens.Count == 0)
            {
                // Старая логика для обратной совместимости
                //ProcessOldLogic(); //For what, it is not able to use
                Debug.Log("No tokenizer or requestTokens is empty!");
                return;
            }
            
            // Новая логика с токенами
            ProcessTokenLogic();
        }
    }
    
    private void ProcessTokenLogic()
    {
        // Сброс нейронов кроме bias
        for (int i = 1; i < neurones.Count; i++)
        {
            neurones[i] = 0f;
        }
        neurones[0] = 1f; // bias
        
        // Определяем текущий токен
        int currentToken;
        if (!isGenerating && currentTokenIndex < requestTokens.Count)
        {
            currentToken = requestTokens[currentTokenIndex];
            currentTokenIndex++;
            
            if (currentToken == 2) // EOS токен
            {
                isGenerating = true;
                nextInputToken = 1; // SOS для начала генерации
            }
        }
        else if (isGenerating)
        {
            currentToken = nextInputToken;
        }
        else
        {
            return;
        }
        
        // One-hot encoding для токена (проверяем доступность)
        int neuronIndex = 1 + currentToken;
        if (neuronIndex < neurones.Count && neuronIndex >= 1)
        {
            // Проверяем, доступен ли этот токен
            if (availableRussianTokens.Contains(currentToken) || currentToken < 4)
            {
                neurones[neuronIndex] = 1f;
            }
            else
            {
                // Если токен недоступен, используем UNK
                neurones[1 + Tokenizer.UNK_TOKEN] = 1f;
            }
        }
        
        // Пропускаем через сеть
        for (int i = 0; i < order.Count; i++)
        {
            int thisNeuron = order[i];
            if (thisNeuron >= outConnections)
            {
                neurones[thisNeuron] = (float)Math.Tanh(neurones[thisNeuron]);
            }
            foreach (var b in adjList[thisNeuron])
            {
                if (b.Key < neurones.Count)
                {
                    neurones[b.Key] += b.Value * neurones[thisNeuron];
                }
            }
        }
        
        // Генерация выхода
    if (isGenerating)
    {
        float maxActivation = -1f;
        int predictedToken = 3; // UNK по умолчанию
        
        for (int i = outConnections; i < Mathf.Min(outConnections + englishVocabSize, neurones.Count); i++)
        {
            if (neurones[i] > maxActivation)
            {
                maxActivation = neurones[i];
                predictedToken = i - outConnections;
            }
        }
        
        // Сохраняем ВСЕ токены, включая SOS и EOS
        // Но! В начале генерации nextInputToken = 1 (SOS), его тоже нужно сохранить
        if (outputTokens.Count == 0 && nextInputToken == 1)
        {
            outputTokens.Add(1); // Добавляем SOS в начале генерации
        }
        
        outputTokens.Add(predictedToken);
        
        // Детокенизируем для отображения (DetokenizeEnglish пропустит SOS/EOS)
        if (tokenizer != null)
        {
            answerNN = tokenizer.DetokenizeEnglish(outputTokens);
        }
        
        // Подготавливаем следующий вход
        nextInputToken = predictedToken;
        
        // Проверка окончания
        if (predictedToken == 2 || // EOS
            outputTokens.Count >= answerTokens.Count + 10 ||
            iterations >= maxIterations)
        {
            FinishTokenGeneration();
        }
    }
        
        // Обновление RNN состояния
        UpdateRNNState();
        iterations++;
    }
    
    private void FinishTokenGeneration()
    {
        go = false;
        isGenerating = false;
        
        // Удаляем EOS если есть
        /*if (outputTokens.Count > 0 && outputTokens.Last() == 2)
        {
            outputTokens.RemoveAt(outputTokens.Count - 1);
        }*/
        
        // Вычисляем fitness
        CalculateTokenFitness();
        
        //++spawnerOfNN.GetComponent<GameManager>().readyToCheck;
        cIterator = 0;
    }
    
    private void CalculateTokenFitness()
    {
        if (answerTokens.Count == 0 || outputTokens.Count == 0)
        {
            howIsGood = 0f;
            return;
        }
        
        // Теперь сравниваем ПОЛНЫЕ последовательности с SOS/EOS
        
        // 1. Точное совпадение последовательностей
        int exactMatches = 0;
        int minLength = Mathf.Min(outputTokens.Count, answerTokens.Count);
        
        for (int i = 0; i < minLength; i++)
        {
            if (outputTokens[i] == answerTokens[i])
                exactMatches++;
        }
        
        float exactMatchScore = answerTokens.Count > 0 ? 
            (float)exactMatches / answerTokens.Count : 0f;
        
        // 2. Совпадение уникальных токенов (исключая служебные)
        var outputWords = new HashSet<int>();
        var answerWords = new HashSet<int>();
        
        foreach (var token in outputTokens)
        {
            if (token > 3) // Исключаем PAD(0), SOS(1), EOS(2), UNK(3)
                outputWords.Add(token);
        }
        
        foreach (var token in answerTokens)
        {
            if (token > 3)
                answerWords.Add(token);
        }
        
        outputWords.IntersectWith(answerWords);
        
        float wordMatchScore = answerWords.Count > 0 ? 
            (float)outputWords.Count / answerWords.Count : 0f;
        
        // 3. Наибольшая общая подпоследовательность
        float lcsScore = CalculateLCS(outputTokens, answerTokens);
        
        // 4. Бонус за правильные SOS/EOS
        float structureBonus = 0f;
        
        // Бонус за правильный SOS в начале
        if (outputTokens.Count > 0 && outputTokens[0] == 1 && answerTokens.Count > 0 && answerTokens[0] == 1)
            structureBonus += 0.03f;
        
        // Бонус за правильный EOS в конце
        if (outputTokens.Count > 0 && outputTokens.Last() == 2 && answerTokens.Count > 0 && answerTokens.Last() == 2)
            structureBonus += 0.03f;
        
        //Нейросети выводили {1, 0, 0, 0...}
        /*for(int i = 0; i != answerTokens.Count; ++i)
        {
            if (outputTokens.Count > i && answerTokens[i] > 3 && outputTokens[i] > 3){
                structureBonus += 0.08f;
            }
        }*/

        if(outputWords.Count == 0){
            structureBonus -= 0.5f;
        }
        float innovationRatio = 0;
        //innovationRatio = inpInnov.Count / (float)availableInputNeurons.Count / 1.5f; //Force adding neurones. NN gets in trouble with local optimum
        float innovationBonus = Mathf.Min(innovationRatio, 1f);
        // 5. Комбинированный fitness (со структурным бонусом)

        float tokenUsageBonus = 0f;
        //int usedAvailableTokens = 0;

        // 4. Штраф за длину * Проверка на проблему
        float penaltyForLength = 0;

        if(outputTokens.Count > answerTokens.Count)
        {
            penaltyForLength -= outputTokens.Count/answerTokens.Count/10.0f;
            //Debug.Log(penaltyForLength);
        }

        /*foreach (var token in outputTokens)
        {
            if (availableEnglishTokens.Contains(token))
                usedAvailableTokens++;
        }
        
        if (outputTokens.Count > 0 && availableEnglishTokens.Count > 4) // Исключаем служебные
        {
            tokenUsageBonus = (float)usedAvailableTokens / outputTokens.Count * 0.1f;
        }*/

        textHowIsGood = Mathf.Min(200.0f, exactMatchScore * 0.5f + 
                    wordMatchScore * 0.35f + 
                    lcsScore * 0.15f +
                    structureBonus +
                    penaltyForLength);

        howIsGood = Mathf.Min(200.0f, textHowIsGood +
                    innovationBonus +
                    tokenUsageBonus);
        
        accumulatedFitness += howIsGood;
        ++processedInBatch;
    }
    
    private float CalculateLCS(List<int> a, List<int> b)
    {
        if (a.Count == 0 || b.Count == 0) return 0f;
        
        int[,] dp = new int[a.Count + 1, b.Count + 1];
        
        for (int i = 1; i <= a.Count; i++)
        {
            for (int j = 1; j <= b.Count; j++)
            {
                if (a[i - 1] == b[j - 1])
                {
                    dp[i, j] = dp[i - 1, j - 1] + 1;
                }
                else
                {
                    dp[i, j] = Mathf.Max(dp[i - 1, j], dp[i, j - 1]);
                }
            }
        }
        
        return (float)dp[a.Count, b.Count] / Mathf.Max(a.Count, b.Count, 1);
    }
    
    private void UpdateRNNState()
    {
        for (int i = 0; i < inpInnov.Count; i++)
        {
            if (actConnect[i] == true && RNNs[i] == true)
            {
                if (outInnov[i] < RNNneurones.Count)
                {
                    RNNneurones[outInnov[i]] += neurones[inpInnov[i]] * weights[i];
                }
            }
        }
        
        // Копируем состояние
        for (int i = 0; i < Mathf.Min(RNNneurones.Count, neurones.Count); i++)
        {
            neurones[i] = RNNneurones[i];
        }
        
        // Затухание
        for (int i = 0; i < RNNneurones.Count; i++)
        {
            RNNneurones[i] *= 0.95f;
        }
    }
    
    public void SetTokenizer(Tokenizer tokenizer, int ruVocabSize, int enVocabSize)
    {
        this.tokenizer = tokenizer;
        this.russianVocabSize = ruVocabSize;
        this.englishVocabSize = enVocabSize;
        
        // Пересчитываем архитектуру сети
        outConnections = 1 + russianVocabSize;
    }
    
    // Остальные методы NEAT остаются без изменений
    public void AddNode()
    {
        int ind = UnityEngine.Random.Range(0, outInnov.Count);
        int reccurency = 0;
        while (reccurency < 5)
        {
            if (RNNs[ind] == false && actConnect[ind] == true)
            {
                break;
            }
            ind = UnityEngine.Random.Range(0, outInnov.Count);
            ++reccurency;
        }
        if (reccurency >= 5)
        {
            return;
        }
        neurones.Add(0);
        adjList.Add(new Dictionary<int, float>());

        actConnect[ind] = false;

        weights.Add(weights[ind]);
        inpInnov.Add(inpInnov[ind]);
        outInnov.Add(neurones.Count - 1);
        RNNs.Add(false);
        actConnect.Add(true);
        RNNneurones.Add(0);
        innovations.Add(spawnerOfNN.GetComponent<GameManager>().DealInnovations(inpInnov[inpInnov.Count - 1], outInnov[inpInnov.Count - 1], RNNs[inpInnov.Count - 1]));

        weights.Add(1f);
        inpInnov.Add(neurones.Count - 1);
        outInnov.Add(outInnov[ind]);
        RNNs.Add(false);
        actConnect.Add(true);
        innovations.Add(spawnerOfNN.GetComponent<GameManager>().DealInnovations(inpInnov[inpInnov.Count - 1], outInnov[inpInnov.Count - 1], RNNs[inpInnov.Count - 1]));
        makeOrder();
    }
    
    public void AddLink()
    {
        bool errorInOut = false;
        weights.Add(UnityEngine.Random.Range(-3f, 3f));
        
        // Выбираем входной нейрон только из доступных
        if (availableInputNeurons.Count == 0)
        {
            Debug.Log("No available input neurons!");
            return;
        }
        inpInnov.Add(availableInputNeurons[UnityEngine.Random.Range(0, availableInputNeurons.Count)]);
        
        List<int> TakenConnections = new List<int>();
        
        // Выбираем выходной нейрон только из доступных
        if (availableOutputNeurons.Count == 0)
        {
            Debug.Log("No available output neurons!");
            return;
        }
        int probableOut = availableOutputNeurons[UnityEngine.Random.Range(0, availableOutputNeurons.Count)];

        for (int i = 0; i < outInnov.Count; i++)
        {
            if (inpInnov[i] == inpInnov[inpInnov.Count - 1])
            {
                TakenConnections.Add(outInnov[i]);
            }
        }
        
        foreach (int a in TakenConnections)
        {
            if (a == probableOut || probableOut == inpInnov.Last())
            {
                errorInOut = true;
                break;
            }
        }
        
        if (probableOut >= neurones.Count)
        {
            Debug.Log("!!!");
            probableOut = neurones.Count - 1;
        }
        
        outInnov.Add(probableOut);
        RNNs.Add(false);
        actConnect.Add(true);
        
        if (errorInOut == true || GenToPh().SequenceEqual(new List<int> { 1, outConnections, 0 }))
        {
            inpInnov.RemoveAt(inpInnov.Count - 1);
            outInnov.RemoveAt(outInnov.Count - 1);
            RNNs.RemoveAt(RNNs.Count - 1);
            actConnect.RemoveAt(actConnect.Count - 1);
            weights.RemoveAt(weights.Count - 1);
            makeOrder();
            ++recursionAddLink;
            if (recursionAddLink < 3)
            {
                AddLink();
                recursionAddLink = 0;
            }
        }
        else
        {
            if (UnityEngine.Random.Range(0, 6) >= 2)
            {
                if (RNNs.Count > 0)
                {
                    RNNs[RNNs.Count - 1] = true;
                    innovations.Add(spawnerOfNN.GetComponent<GameManager>().DealInnovations(inpInnov[inpInnov.Count - 1], outInnov[inpInnov.Count - 1], RNNs[inpInnov.Count - 1]));
                }
            }
            else
            {
                innovations.Add(spawnerOfNN.GetComponent<GameManager>().DealInnovations(inpInnov[inpInnov.Count - 1], outInnov[inpInnov.Count - 1], RNNs[inpInnov.Count - 1]));
            }
        }
        makeOrder();
    }
    
    public List<int> GenToPh()
    {
        List<float> _weights = new List<float>(weights);
        List<int> _inpInnov = new List<int>(inpInnov);
        List<int> _outInnov = new List<int>(outInnov);
        List<bool> _actConnect = new List<bool>(actConnect);
        List<bool> _RNNs = new List<bool>(RNNs);
        List<int> nullConn = new List<int>();
        List<int> inDegree = new List<int>();
        List<int> order1 = new List<int>();
        int neuronesCount = neurones.Count;
        for (int i = 0; i < _actConnect.Count; i++)
        {
            if (_actConnect[i] == false || _RNNs[i] == true)
            {
                _actConnect.RemoveAt(i);
                _weights.RemoveAt(i);
                _inpInnov.RemoveAt(i);
                _outInnov.RemoveAt(i);
                _RNNs.RemoveAt(i);
                --i;
            }
        }
        adjList.Clear();
        for (int i = 0; i < neurones.Count; i++)
        {
            adjList.Add(new Dictionary<int, float>());
        }
        for (int i = 0; i < _inpInnov.Count; i++)
        {
            adjList[_inpInnov[i]].Add(_outInnov[i], _weights[i]);
        }
        for (int i = 0; i < adjList.Count; i++)
        {
            inDegree.Add(0);
        }
        for (int i = 0; i < neurones.Count; i++)
        {
            foreach (var b in adjList[i])
            {
                ++inDegree[b.Key];
            }
        }
        for (int i = 0; i < inDegree.Count; i++)
        {
            if (inDegree[i] == 0)
            {
                nullConn.Add(i);
            }
        }
        for (int i = 0; i != nullConn.Count; ++i)
        {
            int ie = nullConn[i];
            order1.Add(ie);
            foreach (var b in adjList[ie])
            {
                int a = b.Key;
                --inDegree[b.Key];
                if (inDegree[a] == 0)
                {
                    nullConn.Add(b.Key);
                }
            }
        }
        if (order1.Count != neuronesCount)
        {
            return new List<int> { 1, outConnections, 0 };
        }
        if (!order1.Any())
        {
            Debug.Log("Empty");
            return new List<int> { 1, outConnections, 0 };
        }
        return order1;
    }
    
    public void makeOrder()
    {
        List<int> a = GenToPh();
        order = a;
    }
    
    public void Adder()
    {
        adjList.Clear();
        for (int i = 0; i < neurones.Count; i++)
        {
            adjList.Add(new Dictionary<int, float>());
        }
        if (addition == 1)
        {
            AddLink();
        }
        if (addition == 2)
        {
            AddNode();
        }
        makeOrder();
        addition = 0;
    }

    public bool correctGen(List<int> a)
    {
        List<int> genes = new List<int>(a);
        bool yes = true;
        if(a.SequenceEqual(new List<int> {1, outConnections, 0})){
            return false;
        }
        if (genes[0] == 0 || genes[0] > outConnections)
        {
            genes.RemoveRange(0, genes.Count - 3);
            for (int i = 1; i != outConnections + 1; ++i)
            {
                if (genes.Contains(i))
                {
                    yes = false;
                    break;
                }
            }
        }
        return yes;
    }

    public void UpdateAvailableTokens(HashSet<int> ruTokens, HashSet<int> enTokens)
    {
        availableRussianTokens = new HashSet<int>(ruTokens);
        availableEnglishTokens = new HashSet<int>(enTokens);
        UpdateAvailableNeurons();
    }

    // Метод для обновления доступных нейронов:
    private void UpdateAvailableNeurons()
    {
        availableInputNeurons.Clear();
        availableOutputNeurons.Clear();
        
        // 1. Всегда добавляем bias нейрон (индекс 0) для входов
        availableInputNeurons.Add(0); // bias
        
        // 2. Входные нейроны (русские токены) - ВСЕ доступные, включая служебные
        foreach (var token in availableRussianTokens)
        {
            // Индекс нейрона для русского токена: 1 + token
            // (bias=0, токен0=1, токен1=2, токен2=3, токен3=4, токен4=5...)
            int neuronIndex = 1 + token; // +1 для смещения из-за bias
            if (neuronIndex < neurones.Count && !availableInputNeurons.Contains(neuronIndex))
            {
                availableInputNeurons.Add(neuronIndex);
            }
        }
        
        // 3. Выходные нейроны (английские токены) - ВСЕ доступные, включая служебные
        foreach (var token in availableEnglishTokens)
        {
            // Индекс нейрона для английского токена: outConnections + token
            // outConnections = 1 + russianVocabSize
            int neuronIndex = outConnections + token;
            if (neuronIndex < neurones.Count && !availableOutputNeurons.Contains(neuronIndex))
            {
                availableOutputNeurons.Add(neuronIndex);
            }
        }
        
        // 4. ВСЕГДА добавляем служебные токены для выхода, даже если их нет в availableEnglishTokens
        // Это критически важно для генерации EOS!
        for (int serviceToken = 0; serviceToken <= 3; serviceToken++)
        {
            int neuronIndex = outConnections + serviceToken;
            if (neuronIndex < neurones.Count && !availableOutputNeurons.Contains(neuronIndex))
            {
                availableOutputNeurons.Add(neuronIndex);
            }
        }
        
        // 5. Все скрытые нейроны (добавленные через AddNode) всегда доступны
        int hiddenStart = 1 + russianVocabSize + englishVocabSize;
        for (int i = hiddenStart; i < neurones.Count; i++)
        {
            if (!availableInputNeurons.Contains(i))
                availableInputNeurons.Add(i);
            if (!availableOutputNeurons.Contains(i))
                availableOutputNeurons.Add(i);
        }
        
        // Отладка
        /*Debug.Log($"Available output neurons: {availableOutputNeurons.Count} total");
        if (availableOutputNeurons.Count > 0)
        {
            Debug.Log($"First few output neurons: {string.Join(", ", availableOutputNeurons.Take(10))}");
        }*/
    }

    public void ResetForNewPhase()
    {
        go = true;
        isGenerating = false;
        howIsGood = 0;
        textHowIsGood = 0;
        cIterator = 0;
        currentTokenIndex = 0;
        nextInputToken = 1;
        answerNN = "";
        outputTokens.Clear();
        iterations = 0;
        
        // Сбрасываем состояния нейронов
        for (int i = 0; i < neurones.Count; i++)
        {
            neurones[i] = 0f;
            RNNneurones[i] = 0f;
        }
        neurones[0] = 1f;
        RNNneurones[0] = 1f;
        
        UpdateAvailableNeurons();
    }

    public float GetBatchAverageFitness()
    {
        if (processedInBatch == 0) return 0f;
        return accumulatedFitness / processedInBatch;
    }
    
    // Добавьте метод сброса для нового батча
    public void ResetBatchStats()
    {
        accumulatedFitness = 0f;
        processedInBatch = 0;
    }

    public int getInitalNeurones()
    {
        return initalNeurones;
    }

    public int getOutConnections()
    {
        return outConnections;
    }
}