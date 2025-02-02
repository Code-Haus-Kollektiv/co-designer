using System;

using Grasshopper.Kernel;

using Newtonsoft.Json;
using Codesigner.Models;
using Grasshopper;
using System.Linq;
using OpenAI.Chat;


namespace Codesigner
{
  public class GenerateScript : GH_Component
  {
    /// <summary>
    /// Each implementation of GH_Component must provide a public 
    /// constructor without any arguments.
    /// Category represents the Tab in which the component will appear, 
    /// Subcategory the panel. If you use non-existing tab or panel names, 
    /// new tabs/panels will automatically be created.
    /// </summary>
    public GenerateScript()
      : base("OpenAI Component", "Nickname",
        "Description of component",
        "Category", "Subcategory")
    {
    }

    /// <summary>
    /// Registers all the input parameters for this component.
    /// </summary>
    protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
    {
      pManager.AddTextParameter("ApiKey", "AK", "OpenAPI key", GH_ParamAccess.item);
      pManager.AddTextParameter("Message", "M", "Prompt", GH_ParamAccess.item);
      pManager.AddBooleanParameter("Run", "R", "Run", GH_ParamAccess.item);
    }

    /// <summary>
    /// Registers all the output parameters for this component.
    /// </summary>
    protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
    {
      pManager.AddTextParameter("Response", "R", "Prompt response", GH_ParamAccess.item);
    }

    /// <summary>
    /// This is the method that actually does the work.
    /// </summary>
    /// <param name="DA">The DA object can be used to retrieve data from input parameters and 
    /// to store data in output parameters.</param>
    protected override void SolveInstance(IGH_DataAccess DA)
    {
      string key = "";
      string message = "";
      bool run = false;
      if (!DA.GetData(0, ref key)) { return; }
      if (!DA.GetData(1, ref message)) { return; }
      DA.GetData(2, ref run);

      var client = new GrasshopperAIClient(key);
      if (run)
      {
        var schema = client.GenerateGrasshopper<GrasshopperSchema>(message);


        var doc = this.OnPingDocument();
        foreach (var component in schema.Components)
        {
          InstantiateComponent(doc, component);
        }


        DA.SetData(0, JsonConvert.SerializeObject(schema));
      }

    }

    public static void InstantiateComponent(GH_Document doc, Component addition)
    {
      try
      {
        string name = addition.Name;
        IGH_ObjectProxy myProxy = GetObject(name);
        if (myProxy is null)
          return;

        Guid myId = myProxy.Guid;

        var emit = Instances.ComponentServer.EmitObject(myId);

        doc.AddObject(emit, false);
        emit.Attributes.Pivot = new System.Drawing.PointF(x: (float)addition.Position.X, y: (float)addition.Position.Y);
      }
      catch
      {
      }
    }

    private static IGH_ObjectProxy GetObject(string name)
    {
      IGH_ObjectProxy[] results = Array.Empty<IGH_ObjectProxy>();
      double[] resultWeights = new double[] { 0 };
      Instances.ComponentServer.FindObjects(new string[] { name }, 10, ref results, ref resultWeights);

      var myProxies = results.Where(ghpo => ghpo.Kind == GH_ObjectType.CompiledObject);

      var _components = myProxies.OfType<IGH_Component>();
      var _params = myProxies.OfType<IGH_Param>();

      // Prefer Components to Params
      var myProxy = myProxies.First();
      if (_components != null)
        myProxy = _components.FirstOrDefault() as IGH_ObjectProxy;
      else if (myProxy != null)
        myProxy = _params.FirstOrDefault() as IGH_ObjectProxy;

      myProxy = Instances.ComponentServer.FindObjectByName(name, true, true);

      return myProxy;
    }

    /// <summary>
    /// Provides an Icon for every component that will be visible in the User Interface.
    /// Icons need to be 24x24 pixels.
    /// You can add image files to your project resources and access them like this:
    /// return Resources.IconForThisComponent;
    /// </summary>
    protected override System.Drawing.Bitmap Icon => null;

    /// <summary>
    /// Each component must have a unique Guid to identify it. 
    /// It is vital this Guid doesn't change otherwise old ghx files 
    /// that use the old ID will partially fail during loading.
    /// </summary>
    public override Guid ComponentGuid => new Guid("0b6ba781-c7e4-48bb-895d-72180ad81be0");
  }
}