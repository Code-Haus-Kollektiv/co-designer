using System;
using System.Collections.Generic;
using Codesigner.Models;
using Codesigner.Structured;
using Newtonsoft.Json;
using OpenAI.Chat;

namespace Codesigner
{
    public class GrasshopperAIClient
    {
        private readonly ChatClient _client;
        private readonly ChatCompletionOptions _options;
        public GrasshopperAIClient(string apiKey, string model = "gpt-4o")
        {
            _client = new ChatClient(model: model, apiKey: apiKey);
            _options = new ChatCompletionOptions()
            {
                ResponseFormat = StructuredOutputsExtensions.CreateJsonSchemaFormat<GrasshopperSchema>("grasshopper_components", jsonSchemaIsStrict: true)
            };

        }

        public T GenerateGrasshopper<T>(string message)
        {
            List<ChatMessage> messages = new List<ChatMessage>()
            {
                new SystemChatMessage(@"
                You are a Grasshopper expert. Your task is to generate structured JSON output that will be used to create Grasshopper files. Your output must strictly conform to the structured JSON schema provided in the system.
Instructions:

Generate only valid JSON that exactly matches the schema provided.
Include all required fields with correct data types.
Do not include any extra text, commentary, or formatting outside of the JSON.
If no components or connections are required, output empty arrays accordingly.
Use your expertise in Grasshopper to accurately describe components, ports, and connections based on the user's instructions.
Use your expertise in Grasshopper to give the parameters the correct names that are also used in Grasshopper.
Your output will be directly used to create a valid Grasshopper file, so accuracy and strict adherence to the schema are essential."),
                new UserChatMessage(message)
            };
            var response = _client.CompleteChat(messages, _options);
            var text = response.Value.Content[0].Text;

            return JsonConvert.DeserializeObject<T>(text);
        }
    }
}