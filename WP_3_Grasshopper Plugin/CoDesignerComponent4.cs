using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using Grasshopper;
using Grasshopper.GUI;
using Grasshopper.GUI.Canvas;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Attributes;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Rhino;

public class CoDesignONNX : GH_Component
{
    // This value will be updated via SolveInstance and passed to the inference method.
    private string _inputGuid = "5b850221-b527-4bd6-8c62-e94168cd6efa";
    
    // Cached inference session and JSON data to avoid reloading for every inference.
    private InferenceSession _session;
    private Dictionary<int, string> _indexToLabel;
    private List<string> _guidEncoder;
    private List<string> _nameEncoder;

    public CoDesignONNX()
      : base("Auto Instantiate Component", "AutoInst",
             "Auto Instantiate Component based on ONNX model inference using the clicked component's name and GUID",
             "chk", "co-designer")
    {
    }

    public override Guid ComponentGuid => new Guid("D5124C02-12E6-4A80-A875-C303B87C38D9");

    public override void AddedToDocument(GH_Document document)
    {
        base.AddedToDocument(document);
        Instances.ActiveCanvas.DocumentObjectMouseDown += OnDocumentChanged;
        Instances.ActiveCanvas.KeyDown += OnKeyDown;
        // Initialize the ONNX session and load JSON configuration once.
        InitializeSession();
    }

    public override void RemovedFromDocument(GH_Document document)
    {
        base.RemovedFromDocument(document);
        Instances.ActiveCanvas.DocumentObjectMouseDown -= OnDocumentChanged;
        Instances.ActiveCanvas.KeyDown -= OnKeyDown;
        DisposeSession();
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddTextParameter("Input GUID", "GUID", "Input GUID for model (if needed)", GH_ParamAccess.item, _inputGuid);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        // No outputs are required.
    }

    protected override void SolveInstance(IGH_DataAccess da)
    {
        da.GetData(0, ref _inputGuid);
    }

    private void OnDocumentChanged(object sender, GH_CanvasObjectMouseDownEventArgs e)
    {
        if (e.Document == null) return;
        var selectedComponents = e.Document.ActiveObjects()
            .Where(obj => obj.Attributes.Selected)
            .OfType<GH_Component>()
            .ToList();
        if (!selectedComponents.Any()) return;
        foreach (var component in selectedComponents)
        {
            HandleComponentSelected(component);
        }
    }

    /// <summary>
    /// When a component is clicked (and selected), run inference on its name and GUID.
    /// If a valid target component GUID is returned, instantiate it next to the clicked component.
    /// </summary>
    private void HandleComponentSelected(GH_Component selectedComponent)
    {
        if (selectedComponent.ComponentGuid == this.ComponentGuid)
            return;

        string clickedName = selectedComponent.Name;
        string clickedGuid = selectedComponent.ComponentGuid.ToString();
        RhinoApp.WriteLine($"[Debug] Running inference for component: {clickedName}, GUID: {clickedGuid}");

        // Run cached model inference
        string resultGuidString = RunModelInference(clickedName, clickedGuid);
        if (string.IsNullOrEmpty(resultGuidString))
        {
            RhinoApp.WriteLine("[Debug] Model returned no valid component for: " + clickedName);
            return;
        }
        if (!Guid.TryParse(resultGuidString, out Guid resultGuid))
        {
            RhinoApp.WriteLine("[Debug] Model returned an invalid GUID: " + resultGuidString);
            return;
        }
        var newComponent = Grasshopper.Instances.ComponentServer.EmitObject(resultGuid) as GH_Component;
        if (newComponent != null)
        {
            var attributes = new CustomAttributesOnnx(newComponent);
            newComponent.Attributes = attributes;
            PointF newLocation = selectedComponent.Attributes.Pivot;
            newLocation.X += selectedComponent.Attributes.Bounds.Width + 10; // slight offset
            newComponent.Attributes.Pivot = newLocation;
            Instances.ActiveDocument.AddObject(newComponent, false);

            // Optionally connect outputs to inputs.
            if (selectedComponent.Params.Output.Count > 0 && newComponent.Params.Input.Count > 0)
            {
                var sourceOutput = selectedComponent.Params.Output[0];
                var targetInput = newComponent.Params.Input[0];
                targetInput.AddSource(sourceOutput);
                RhinoApp.WriteLine("[Debug] Connected source output to target input.");
            }

            selectedComponent.ExpireSolution(true);
            newComponent.ExpireSolution(true);
        }
        else
        {
            RhinoApp.WriteLine("[Debug] Component with GUID " + resultGuidString + " does not exist in the component server.");
        }
    }

    private void OnKeyDown(object sender, KeyEventArgs e)
    {
        if (e.KeyCode == Keys.Escape)
        {
            var lastAdded = Instances.ActiveDocument.Objects
                .OfType<GH_Component>()
                .LastOrDefault(c => c.Attributes is CustomAttributesOnnx attributes && attributes.IsNewComponent);
            if (lastAdded != null)
            {
                Instances.ActiveDocument.RemoveObject(lastAdded, true);
                RhinoApp.WriteLine("[Debug] Removed last added component via Escape key.");
            }
        }
    }

    /// <summary>
    /// Initializes the ONNX inference session and loads the configuration files.
    /// </summary>
    private void InitializeSession()
    {
        try
        {
            // *** Configure paths to your model and JSON configuration files ***
            string modelPath = @"./Resources/xgboost_model.onnx";
            string labelMappingPath = @"./Resources/index_to_label.json";
            string guidEncoderPath = @"./Resources/CurrentGUID_encoder.json";
            string nameEncoderPath = @"./Resources/CurrentName_encoder.json";

            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"Model file not found at path: {modelPath}");
            if (!File.Exists(labelMappingPath))
                throw new FileNotFoundException($"Label mapping file not found at path: {labelMappingPath}");
            if (!File.Exists(guidEncoderPath))
                throw new FileNotFoundException($"GUID encoder file not found at path: {guidEncoderPath}");
            if (!File.Exists(nameEncoderPath))
                throw new FileNotFoundException($"Name encoder file not found at path: {nameEncoderPath}");

            // Load and cache JSON configurations.
            _indexToLabel = JsonConvert.DeserializeObject<Dictionary<int, string>>(File.ReadAllText(labelMappingPath));
            _guidEncoder = ParseEncoderJson(File.ReadAllText(guidEncoderPath));
            _nameEncoder = ParseEncoderJson(File.ReadAllText(nameEncoderPath));

            // Use CPU execution provider; change as needed.
            var options = new SessionOptions();
            _session = new InferenceSession(modelPath, options);
            RhinoApp.WriteLine("[Debug] Inference session initialized successfully.");
        }
        catch (Exception ex)
        {
            RhinoApp.WriteLine("[Error] Initializing inference session failed: " + ex.Message);
        }
    }

    /// <summary>
    /// Disposes the cached inference session.
    /// </summary>
    private void DisposeSession()
    {
        if (_session != null)
        {
            _session.Dispose();
            _session = null;
            RhinoApp.WriteLine("[Debug] Inference session disposed.");
        }
    }

    /// <summary>
    /// Runs model inference using cached session and configuration data.
    /// </summary>
    /// <param name="componentName">Name of the clicked component.</param>
    /// <param name="componentGuid">GUID of the clicked component.</param>
    /// <returns>String representing the target component GUID, or empty string on error.</returns>
    private string RunModelInference(string componentName, string componentGuid)
    {
        if (_session == null)
        {
            RhinoApp.WriteLine("[Error] Inference session is not initialized.");
            return "";
        }

        try
        {
            // Encode inputs using cached encoders.
            int encodedGuid = _guidEncoder.IndexOf(componentGuid);
            if (encodedGuid < 0)
            {
                RhinoApp.WriteLine($"[Debug] GUID '{componentGuid}' not found in encoder. Using -1.");
                encodedGuid = -1;
            }
            int encodedName = _nameEncoder.IndexOf(componentName);
            if (encodedName < 0)
            {
                RhinoApp.WriteLine($"[Debug] Name '{componentName}' not found in encoder. Using -1.");
                encodedName = -1;
            }

            // Prepare input tensor.
            // The model expects a tensor with shape [1, 305]
            // Here we fill a 305-element vector with zeros and assign the first two entries.
            float[] inputVector = new float[305];
            inputVector[0] = (float)encodedGuid;
            inputVector[1] = (float)encodedName;
            var tensor = new DenseTensor<float>(inputVector, new int[] { 1, 305 });
            
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", tensor)
            };

            using (var results = _session.Run(inputs))
            {
                var resultsList = results.ToList();
                int outputIndex = resultsList.Count > 1 ? 1 : 0;
                var outputTensor = resultsList[outputIndex].AsTensor<float>();
                float[] outputArray = outputTensor.ToArray();
                int predictedIndex = Array.IndexOf(outputArray, outputArray.Max());

                if (_indexToLabel.ContainsKey(predictedIndex))
                    return _indexToLabel[predictedIndex];
                else
                {
                    RhinoApp.WriteLine("[Debug] Predicted index not found in mapping: " + predictedIndex);
                    return "";
                }
            }
        }
        catch (Exception ex)
        {
            RhinoApp.WriteLine("[Error] Model inference failed: " + ex.Message);
            return "";
        }
    }

    /// <summary>
    /// Helper method to parse an encoder JSON string.
    /// If the JSON is an object with a "classes_" key, returns its list; otherwise, assumes it is a direct list.
    /// </summary>
    private List<string> ParseEncoderJson(string json)
    {
        try
        {
            JToken token = JToken.Parse(json);
            if (token.Type == JTokenType.Object && token["classes_"] != null)
            {
                return token["classes_"].ToObject<List<string>>();
            }
            else if (token.Type == JTokenType.Array)
            {
                return token.ToObject<List<string>>();
            }
            else
            {
                throw new Exception("Unsupported encoder JSON format.");
            }
        }
        catch (Exception ex)
        {
            throw new Exception("Error parsing encoder JSON: " + ex.Message);
        }
    }
}

/// <summary>
/// Custom attributes for this component which allow a custom (pink) background
/// to visually indicate newly instantiated components.
/// </summary>
public class CustomAttributesOnnx : GH_ComponentAttributes
{
    public bool IsNewComponent { get; set; } = false;
    public Color? BackgroundColor { get; set; } = null;

    public CustomAttributesOnnx(IGH_Component component)
        : base(component)
    {
    }

    protected override void Render(GH_Canvas canvas, Graphics graphics, GH_CanvasChannel channel)
    {
        if (channel == GH_CanvasChannel.Objects && IsNewComponent)
        {
            graphics.FillRectangle(Brushes.Pink, Bounds);
        }
        base.Render(canvas, graphics, channel);
    }
}