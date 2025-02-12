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
          GrasshopperUtils.InstantiateComponent(doc, this, component);
        }

        foreach (var connection in schema.Connections)
        {
          GrasshopperUtils.ConnectComponent(doc, schema, connection);
        }


        DA.SetData(0, JsonConvert.SerializeObject(schema));
      }

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