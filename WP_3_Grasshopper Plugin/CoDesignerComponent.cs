using Grasshopper;
using Grasshopper.Kernel;
using System;
using System.Drawing;
using System.Net.Http;
using System.Threading.Tasks;
using System.Linq;
using System.Collections.Generic;
using AutoGen.Core;
using AutoGen.Ollama;
using AutoGen.Ollama.Extension;
using AutoGen.DotnetInteractive; // <-- Make sure you've added the AutoGen.DotnetInteractive package
using Eto.Drawing;
using Google.Protobuf.Compiler;
using Rhino;

namespace CoDesigner.GrasshopperComponents
{
    public class OllamaAgentComponent : GH_Component
    {
        private static readonly HttpClient HttpClient = new HttpClient
        {
            BaseAddress = new Uri("http://localhost:11434")
        };

        public OllamaAgentComponent()
            : base("OllamaAgent", 
                "Ollama",
                "Integrates with an Ollama Agent to send and receive messages, and optionally run code snippets.",
                "Utilities",
                "co-designer")
        {
        }

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddTextParameter("Input Message", "Msg", "Message to send to the Ollama Agent.",
                GH_ParamAccess.item,
                "Please place MyCustomComponent on the canvas and connect it to the existing Area component.");
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddTextParameter("Response", "Res", "Response from the Ollama Agent.", GH_ParamAccess.item);
        }

        protected override async void SolveInstance(IGH_DataAccess DA)
        {
            string filePath;

            if (RhinoDoc.ActiveDoc.Path != null)
            {
                filePath = RhinoDoc.ActiveDoc.Path;
            }
            else
            {
                filePath = "File not saved.";
            }

            var kernel = DotnetInteractiveKernelBuilder
                .CreateKernelBuilder(filePath);

            kernel.BuildAsync();

            string inputMessage = string.Empty;
            if (!DA.GetData(0, ref inputMessage))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Failed to retrieve input message.");
                return;
            }

            // Send the message to the LLM and get the response
            string responseMessage;
            try
            {
                responseMessage = await InteractWithOllamaAgent(inputMessage);
            }
            catch (Exception ex)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"Error communicating with agent: {ex.Message}");
                responseMessage = "Error";
            }


            // Output the final message
            DA.SetData(0, responseMessage);
        }

        private async Task<string> InteractWithOllamaAgent(string message)
        {
            HttpClient httpClient = new HttpClient
            {
                BaseAddress = new Uri("http://localhost:11434")
            };

            // Create your OllamaAgent, with the possibility to run code snippet
            var ollamaAgent = new OllamaAgent(
                    httpClient: httpClient,
                    name: "ollama",
                    modelName: "llama3.2:latest",
                    systemMessage: "You are a helpful AI assistant that can generate C# code if needed.")
                .RegisterMessageConnector()
                .RegisterPrintMessage();

            // You could also add your own custom “middleware” to process snippet generation here
            // But for simplicity, we will just get the text reply back:
            var reply = await ollamaAgent.SendAsync(message);
            return reply.GetContent();
        }

        // ----------------------------------------------------------
        // 3) Helper to extract C# code between ```csharp ... ``` fences
        // ----------------------------------------------------------
        private string ExtractCsharpCodeSnippet(string content)
        {
            const string startTag = "```csharp";
            const string endTag = "```";
            int startIndex = content.IndexOf(startTag, StringComparison.OrdinalIgnoreCase);
            if (startIndex >= 0)
            {
                int endIndex = content.IndexOf(endTag, startIndex + startTag.Length,
                    StringComparison.OrdinalIgnoreCase);
                if (endIndex >= 0)
                {
                    int codeStart = startIndex + startTag.Length;
                    return content.Substring(codeStart, endIndex - codeStart).Trim();
                }
            }

            return null;
        }

        public override GH_Exposure Exposure => GH_Exposure.primary;

        public override Guid ComponentGuid => new Guid("12345678-90AB-CDEF-1234-567890ABCDEF");

        // protected override Bitmap Icon => null;
    }
}