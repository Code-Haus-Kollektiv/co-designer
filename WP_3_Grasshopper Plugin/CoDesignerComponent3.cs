// using Grasshopper.GUI.Canvas;
// using Grasshopper.GUI;
// using Grasshopper.Kernel.Attributes;
// using Grasshopper.Kernel;
// using Grasshopper;
// using System;
// using System.Drawing;
// using System.Linq;
// using System.Windows.Forms;
// using System.Collections.Generic;
// using System.IO;
// using System.Reflection;
//
// // ONNX Runtime
// using Microsoft.ML.OnnxRuntime;
// using Microsoft.ML.OnnxRuntime.Tensors;
// // For JSON (Newtonsoft or System.Text.Json)
// using Newtonsoft.Json;
//
// public class codesigner : GH_Component
// {
//     private GH_Document ghDocument;
//
//     // ONNX session
//     private InferenceSession onnxSession;
//
//     // Map from class index => "guid|Name"
//     private Dictionary<int, string> indexToLabelMap;
//
//     public codesigner()
//       : base("co-designer",
//           "cody",
//              "Predicts the next component using an ONNX model and auto-instantiates it.",
//              "chk", "co-designer")
//     {
//         // 1) Load resources at construction time
//         try
//         {
//             Assembly asm = Assembly.GetExecutingAssembly();
//
//             // Load the ONNX from embedded resource:
//             // The resource name depends on your namespace/folder structure. 
//             // If your .csproj name is "MyGhPlugin", 
//             // and files are "Resources.xgboost_model.onnx",
//             // the resource might be "MyGhPlugin.Resources.xgboost_model.onnx"
//             string onnxResourceName = "CoDesigner.Resources.xgboost_model.onnx";
//             using(Stream onnxStream = asm.GetManifestResourceStream(onnxResourceName))
//             {
//                 if(onnxStream == null)
//                     throw new Exception("Cannot find embedded xgboost_model.onnx");
//                 
//                 // read bytes
//                 byte[] onnxBytes;
//                 using(var ms = new MemoryStream())
//                 {
//                     onnxStream.CopyTo(ms);
//                     onnxBytes = ms.ToArray();
//                 }
//                 // Create the InferenceSession from bytes
//                 onnxSession = new InferenceSession(onnxBytes);
//             }
//
//             // Load the index_to_label.json from embedded resource
//             string mapResourceName = "CoDesigner.Resources.index_to_label.json";
//             using(Stream mapStream = asm.GetManifestResourceStream(mapResourceName))
//             {
//                 if(mapStream == null)
//                     throw new Exception("Cannot find embedded index_to_label.json");
//
//                 using(var reader = new StreamReader(mapStream))
//                 {
//                     string json = reader.ReadToEnd();
//                     indexToLabelMap = JsonConvert.DeserializeObject<Dictionary<int, string>>(json);
//                 }
//             }
//         }
//         catch(Exception ex)
//         {
//             this.AddRuntimeMessage(GH_RuntimeMessageLevel.Error,
//                 "Error loading ONNX or index_to_label resources: " + ex.Message);
//         }
//     }
//
//     public override Guid ComponentGuid => new Guid("0bc0a371-bd1c-486a-8cf9-01c572b71bff");
//
//     public override void AddedToDocument(GH_Document document)
//     {
//         base.AddedToDocument(document);
//         ghDocument = document;
//
//         Instances.ActiveCanvas.DocumentObjectMouseDown += OnDocumentChanged;
//         Instances.ActiveCanvas.KeyDown += OnKeyDown;
//     }
//
//     private void OnDocumentChanged(object sender, GH_CanvasObjectMouseDownEventArgs e)
//     {
//         if (e.Document == null)
//             return;
//
//         var objects = e.Document.ActiveObjects();
//         var selected = objects
//             .OfType<GH_Component>()
//             .Where(c => c.Attributes.Selected)
//             .ToList();
//
//         if (!selected.Any()) return;
//
//         foreach (var comp in selected)
//         {
//             if (comp.ComponentGuid == this.ComponentGuid) 
//                 continue; // skip self
//             HandleComponentSelected(comp);
//         }
//     }
//
//     private void HandleComponentSelected(GH_Component selectedComponent)
//     {
//         if (onnxSession == null || indexToLabelMap == null)
//         {
//             this.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
//                 "ONNX session or indexToLabel map not loaded. Cannot predict next component.");
//             return;
//         }
//
//         // 2) Build features from 'selectedComponent' (this must match your Python logic!)
//         float[] inputData = ExtractFeatures(selectedComponent);
//
//         // 3) ONNX Inference => predicted class index
//         int predIndex = PredictClassIndex(inputData);
//         if (!indexToLabelMap.ContainsKey(predIndex))
//         {
//             this.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
//                 $"Predicted index {predIndex} not in indexToLabel map");
//             return;
//         }
//         string labelStr = indexToLabelMap[predIndex]; // e.g. "abc123|SomeName"
//         string[] parts = labelStr.Split('|');
//         string guidStr = (parts.Length > 0) ? parts[0] : "";
//         // Optionally parse the name = parts[1], if you want
//
//         // 4) Instantiate the predicted component
//         if(Guid.TryParse(guidStr, out Guid nextCompGuid))
//         {
//             var newComp = Instances.ComponentServer.EmitObject(nextCompGuid) as GH_Component;
//             if(newComp == null)
//             {
//                 this.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
//                     $"Failed to instantiate next component: {guidStr}");
//                 return;
//             }
//
//             // Pink attributes
//             var attrs = new CustomAttributes2(newComp);
//             attrs.IsNewComponent = true;
//             newComp.Attributes = attrs;
//
//             // Place it near the selected
//             PointF newLocation = selectedComponent.Attributes.Pivot;
//             newLocation.X += newComp.Attributes.Bounds.Width;
//             newComp.Attributes.Pivot = newLocation;
//
//             Instances.ActiveDocument.AddObject(newComp, false);
//
//             // If you want to connect the first output to first input
//             if(newComp.Params.Input.Count > 0 && selectedComponent.Params.Output.Count > 0)
//             {
//                 var sourceOutput = selectedComponent.Params.Output[0];
//                 var targetInput = newComp.Params.Input[0];
//                 targetInput.AddSource(sourceOutput);
//             }
//
//             selectedComponent.ExpireSolution(true);
//             newComp.ExpireSolution(true);
//         }
//         else
//         {
//             this.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
//                 $"Predicted label does not contain a valid GUID: {guidStr}");
//         }
//     }
//
//     private float[] ExtractFeatures(GH_Component comp)
//     {
//         // Safely cast int->float
//         float numIn     = (float) comp.Params.Input.Count;
//         float numOut    = (float) comp.Params.Output.Count;
//         float numParams = numIn + numOut;
//
//         // If your real model expects, say, 277 features, build that array properly
//         return new float[] { numParams, numIn, numOut, 0.0f };
//     }
//
//     private int PredictClassIndex(float[] inputData)
//     {
//         // Typically your model expects shape [1, featureCount]
//         // so we build a DenseTensor of shape (1, len(inputData))
//         var shape = new int[] {1, inputData.Length};
//         var inputTensor = new DenseTensor<float>(shape);
//         for(int i = 0; i < inputData.Length; i++)
//         {
//             inputTensor[0, i] = inputData[i];
//         }
//
//         // ONNX input name
//         string inputName = onnxSession.InputMetadata.Keys.First();
//         var inputs = new List<NamedOnnxValue>
//         {
//             NamedOnnxValue.CreateFromTensor<float>(inputName, inputTensor)
//         };
//
//         using(var results = onnxSession.Run(inputs))
//         {
//             // If your model's single output is e.g. "probabilities" or "output"
//             // we get a float32 Tensor => shape=(1, num_classes)
//             string outputName = onnxSession.OutputMetadata.Keys.First();
//             var outputTensor = results.First().AsTensor<float>(); // shape = (1, numClasses) or (numClasses)
//             // Argmax
//             float maxVal = float.MinValue;
//             int bestIndex = 0;
//             for(int i=0; i<outputTensor.Length; i++)
//             {
//                 float val = outputTensor[i];
//                 if(val > maxVal)
//                 {
//                     maxVal = val;
//                     bestIndex = i;
//                 }
//             }
//             return bestIndex;
//         }
//     }
//
//     private void OnKeyDown(object sender, KeyEventArgs e)
//     {
//         // Press ESC => remove last pink component
//         if(e.KeyCode == Keys.Escape)
//         {
//             var lastAdded = Instances.ActiveDocument.Objects
//                 .OfType<GH_Component>()
//                 .LastOrDefault(c => c.Attributes is CustomAttributes2 attrs && attrs.IsNewComponent);
//             if (lastAdded != null)
//             {
//                 Instances.ActiveDocument.RemoveObject(lastAdded, true);
//             }
//         }
//     }
//
//     public override void RemovedFromDocument(GH_Document document)
//     {
//         base.RemovedFromDocument(document);
//         Instances.ActiveCanvas.DocumentObjectMouseDown -= OnDocumentChanged;
//         Instances.ActiveCanvas.KeyDown -= OnKeyDown;
//     }
//
//     protected override void RegisterInputParams(GH_InputParamManager pManager) {}
//     protected override void RegisterOutputParams(GH_OutputParamManager pManager) {}
//     protected override void SolveInstance(IGH_DataAccess DA) {}
// }
//
// // Pink background
// public class CustomAttributes2 : GH_ComponentAttributes
// {
//     public bool IsNewComponent { get; set; } = false;
//
//     public CustomAttributes2(IGH_Component comp) : base(comp) {}
//
//     protected override void Render(GH_Canvas canvas, Graphics graphics, GH_CanvasChannel channel)
//     {
//         if(channel == GH_CanvasChannel.Objects)
//         {
//             if(IsNewComponent)
//                 graphics.FillRectangle(Brushes.Pink, Bounds);
//
//             base.Render(canvas, graphics, channel);
//         }
//         else
//         {
//             base.Render(canvas, graphics, channel);
//         }
//     }
// }
