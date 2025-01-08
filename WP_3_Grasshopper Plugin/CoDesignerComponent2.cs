using Grasshopper.GUI.Canvas;
using Grasshopper.GUI;
using Grasshopper.Kernel.Attributes;
using Grasshopper.Kernel;
using Grasshopper;
using System;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;

public class AutoInstantiateComponent : GH_Component
{
    private GH_Document ghDocument;
    private string guid = Guid.NewGuid().ToString();

    public AutoInstantiateComponent()
        : base(
              "Auto Instantiate Component",
              "AutoInst",
              "Auto Instantiate Component",
              "Utilities",
              "co-designer")
    {
    }

    public override Guid ComponentGuid => new Guid("D5124C02-12E6-4A80-A875-C303B87C38D9");

    public override void AddedToDocument(GH_Document document)
    {
        base.AddedToDocument(document);
        ghDocument = document;

        Instances.ActiveCanvas.DocumentObjectMouseDown += OnDocumentChanged;
        Instances.ActiveCanvas.KeyDown += OnKeyDown;
    }

    private void OnDocumentChanged(object sender, GH_CanvasObjectMouseDownEventArgs e)
    {
        if (e.Document == null)
            return;

        var objects = e.Document.ActiveObjects();
        var selectedComponents = objects
            .Where(obj => obj.Attributes.Selected)
            .Cast<GH_Component>()
            .ToList();

        if (!selectedComponents.Any())
            return;

        foreach (var component in selectedComponents)
        {
            HandleComponentSelected(component);
        }
    }

    private void HandleComponentSelected(GH_Component selectedComponent)
    {
        if (selectedComponent.ComponentGuid == this.ComponentGuid)
            return;

        var newComponent = Grasshopper.Instances.ComponentServer.EmitObject(Guid.Parse(guid)) as GH_Component;

        if (newComponent != null)
        {
            // Set custom attributes with a flag for the pink background
            var attributes = new CustomAttributes(newComponent);
            attributes.IsNewComponent = true;
            newComponent.Attributes = attributes;

            PointF newLocation = selectedComponent.Attributes.Pivot;
            newLocation.X += newComponent.Attributes.Bounds.Width;
            newComponent.Attributes.Pivot = newLocation;

            Instances.ActiveDocument.AddObject(newComponent, false);

            if (newComponent.Params.Input.Count > 0 && selectedComponent.Params.Output.Count > 0)
            {
                var sourceOutput = selectedComponent.Params.Output[0];
                var targetInput = newComponent.Params.Input[0];
                targetInput.AddSource(sourceOutput);
            }

            selectedComponent.ExpireSolution(true);
            newComponent.ExpireSolution(true);
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

    public override void RemovedFromDocument(GH_Document document)
    {
        base.RemovedFromDocument(document);
        Instances.ActiveCanvas.DocumentObjectMouseDown -= OnDocumentChanged;
        Instances.ActiveCanvas.KeyDown -= OnKeyDown;
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddTextParameter("GUID", "GUID", "GUID", GH_ParamAccess.item, "5b850221-b527-4bd6-8c62-e94168cd6efa");
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        // No outputs required for this component.
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        DA.GetData(0, ref guid);
    }
}

public class CustomAttributes : GH_ComponentAttributes
{
    public bool IsNewComponent { get; set; } = false;
    public Color? BackgroundColor { get; set; } = null;

    public CustomAttributes(IGH_Component component)
        : base(component)
    { }

    protected override void Render(GH_Canvas canvas, Graphics graphics, GH_CanvasChannel channel)
    {
        if (channel == GH_CanvasChannel.Objects)
        {
            if (IsNewComponent)
            {
                // Render with a pink background for this component only
                graphics.FillRectangle(Brushes.Pink, Bounds);
            }

            // Call the base Render method for standard rendering logic
            base.Render(canvas, graphics, channel);
        }
        else
        {
            base.Render(canvas, graphics, channel);
        }
    }
}
