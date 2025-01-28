using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using GH_IO.Serialization;
using Grasshopper;
using Grasshopper.GUI;
using Grasshopper.Kernel;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;
using Rhino;

namespace CoDesigner
{
    public class Codesigner : GH_Component
    {
        private GH_Document _ghDocument;
        private InferenceSession _onnxSession;
        private Dictionary<int, string> _indexToLabelMap;
        private string _lastOnnxPath = string.Empty;
        private string _lastJsonPath = string.Empty;

        public Codesigner()
            : base("co-designer", "cody", "Predicts the next component using an ONNX model and auto-instantiates it.", "chk", "co-designer")
        {
        }

        public override Guid ComponentGuid => new Guid("0bc0a371-bd1c-486a-8cf9-01c572b71bff");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddTextParameter("ONNX File Path", "ONNX", "Path to the ONNX model file", GH_ParamAccess.item);
            pManager.AddTextParameter("JSON File Path", "JSON", "Path to the index-to-label JSON file", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddTextParameter("Prediction", "P", "Predicted label", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess da)
        {
            if (!TryGetPaths(da, out var onnxFilePath, out var jsonFilePath))
                return;

            if (onnxFilePath != _lastOnnxPath || jsonFilePath != _lastJsonPath || _onnxSession == null)
            {
                ReloadModel(onnxFilePath, jsonFilePath);
            }

            da.SetData(0, "Model loaded");
        }

        private bool TryGetPaths(IGH_DataAccess da, out string onnxFilePath, out string jsonFilePath)
        {
            onnxFilePath = string.Empty;
            jsonFilePath = string.Empty;

            if (!da.GetData(0, ref onnxFilePath) || string.IsNullOrEmpty(onnxFilePath))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Invalid or empty ONNX file path.");
                return false;
            }

            if (!da.GetData(1, ref jsonFilePath) || string.IsNullOrEmpty(jsonFilePath))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Invalid or empty JSON file path.");
                return false;
            }

            if (!File.Exists(onnxFilePath) || !File.Exists(jsonFilePath))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Invalid file paths.");
                return false;
            }

            return true;
        }

        private void ReloadModel(string onnxFilePath, string jsonFilePath)
        {
            DisposeOnnxSession();
            try
            {
                RhinoApp.WriteLine($"Loading ONNX model from: {onnxFilePath}");
                var options = new SessionOptions();
                options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
                options.EnableProfiling = true;
                try
                {
                    options.AppendExecutionProvider_CUDA(0);
                    RhinoApp.WriteLine("CUDA Execution Provider Enabled");
                }
                catch (DllNotFoundException)
                {
                    RhinoApp.WriteLine("CUDA provider not found, using CPU execution provider instead.");
                    options.AppendExecutionProvider_CPU();
                }
                options.IntraOpNumThreads = 2;
                options.InterOpNumThreads = 1;
                
                using (var cts = new CancellationTokenSource(TimeSpan.FromMinutes(1)))
                {
                    _onnxSession = Task.Run(() => new InferenceSession(onnxFilePath, options), cts.Token).GetAwaiter().GetResult();
                }                
                _indexToLabelMap = JsonConvert.DeserializeObject<Dictionary<int, string>>(File.ReadAllText(jsonFilePath));

                _lastOnnxPath = onnxFilePath;
                _lastJsonPath = jsonFilePath;
                RhinoApp.WriteLine("Loaded index-to-label map successfully.");
            }
            catch (Exception ex)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"Error loading ONNX or JSON: {ex.Message}");
            }
        }

        public override void AddedToDocument(GH_Document document)
        {
            base.AddedToDocument(document);
            _ghDocument = document;
            UnsubscribeFromEvents();
            if (Instances.ActiveCanvas != null)
            {
                Instances.ActiveCanvas.DocumentObjectMouseDown += OnDocumentObjectMouseDown;
                Instances.ActiveCanvas.KeyDown += OnKeyDown;
            }
        }

        public override void RemovedFromDocument(GH_Document document)
        {
            base.RemovedFromDocument(document);
            UnsubscribeFromEvents();
            DisposeOnnxSession();
        }

        private void UnsubscribeFromEvents()
        {
            if (Instances.ActiveCanvas != null)
            {
                Instances.ActiveCanvas.DocumentObjectMouseDown -= OnDocumentObjectMouseDown;
                Instances.ActiveCanvas.KeyDown -= OnKeyDown;
            }
        }

        private void OnDocumentObjectMouseDown(object sender, GH_CanvasObjectMouseDownEventArgs e)
        {
            RhinoApp.WriteLine("DocumentObjectMouseDown event triggered.");
            if (e.Document == null) return;
            var selected = e.Document.ActiveObjects().OfType<GH_Component>().Where(c => c.Attributes.Selected).ToList();
            foreach (var comp in selected)
            {
                if (comp.ComponentGuid == this.ComponentGuid) continue;
                HandleComponentSelected(comp);
            }
        }
        
        private void OnKeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Escape)
            {

                var lastAdded = Instances.ActiveDocument.Objects
                    .OfType<GH_Component>()
                    .Last(c => c.Attributes is CustomAttributes attributes && attributes.IsNewComponent);

                if (lastAdded != null)
                {
                    Instances.ActiveDocument.RemoveObject(lastAdded, true);
                }
            }
        }

        private void HandleComponentSelected(GH_Component selectedComponent)
        {
            float[] features = ExtractFeatures(selectedComponent);
            int predictedIndex = PredictClassIndex(features);

            if (_indexToLabelMap.TryGetValue(predictedIndex, out var label))
            {
                RhinoApp.WriteLine($"Predicted label: {label}");
            }
        }

        private float[] ExtractFeatures(GH_Component comp)
        {
            return new float[] { comp.Params.Input.Count + comp.Params.Output.Count, comp.Params.Input.Count, comp.Params.Output.Count, 0.0f };
        }

        private int PredictClassIndex(float[] inputData)
        {
            if (_onnxSession == null) return -1;
            var inputName = _onnxSession.InputMetadata.Keys.First();
            var inputTensor = new DenseTensor<float>(inputData, new[] { 1, inputData.Length });
            using (var results = _onnxSession.Run(new List<NamedOnnxValue>
                       { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) }))
            {
                return results.First().AsTensor<float>().ToArray().ToList()
                    .IndexOf(results.First().AsTensor<float>().Max());
            }
        }


        private void DisposeOnnxSession()
        {
            _onnxSession?.Dispose();
            _onnxSession = null;
        }
    }
}