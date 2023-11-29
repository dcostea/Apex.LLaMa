using LLama.Common;
using LLama;

//var modelPath = @"c:\temp\llama-2-7b-guanaco-qlora.Q8_0.gguf";
var modelPath = @"c:\temp\nous-hermes-llama2-13b.Q4_0.gguf";

// Load weights into memory
var parameters = new ModelParams(modelPath)
{
    ContextSize = 1024,
    Seed = 1337,
    GpuLayerCount = 5
};
using var weights = LLamaWeights.LoadFromFile(parameters);

using var context = weights.CreateContext(parameters);
var ex = new StatelessExecutor(weights, parameters);

var inferenceParams = new InferenceParams()
{
    Temperature = 0.6f,
    AntiPrompts = new List<string> { "Question:", "#", "Question: ", ".\n" },
    MaxTokens = 500,
    //Grammar = grammarInstance
};

string prompt;

// run the inference in a loop to chat with LLM
while (true)
{
    Console.Write("\nQuestion: ");
    Console.ForegroundColor = ConsoleColor.Green;
    prompt = Console.ReadLine();
    Console.ForegroundColor = ConsoleColor.White;
    Console.Write("Answer: ");
    prompt = $"Question: {prompt?.Trim()} Answer: ";
    await foreach (var text in ex.InferAsync(prompt, inferenceParams))
    {
        Console.Write(text);
    }
}
