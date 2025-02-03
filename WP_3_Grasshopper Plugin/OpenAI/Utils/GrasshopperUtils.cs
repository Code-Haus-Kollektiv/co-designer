using System;
using System.Linq;
using Codesigner.Models;
using Grasshopper;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Special;

namespace Codesigner
{
    public static class GrasshopperUtils
    {
        private static IGH_ObjectProxy GetObject(Component addition)
        {
            IGH_ObjectProxy[] results = Array.Empty<IGH_ObjectProxy>();
            double[] resultWeights = new double[] { 0 };
            Instances.ComponentServer.FindObjects(new string[] { addition.Name }, 20, ref results, ref resultWeights);

            var myProxies = results.Where(ghpo => ghpo.Kind == GH_ObjectType.CompiledObject);
            return myProxies.FirstOrDefault();
        }

        public static void ConnectComponent(GH_Document doc, GrasshopperSchema schema, Connection pairing)
        {
            var toComponent = schema.Components.Find(o => o.Id == pairing.ToComponentId);
            var fromComponent = schema.Components.Find(o => o.Id == pairing.FromComponentId);

            if (toComponent is null || fromComponent is null) { return; }

            var toGhComponent = doc.FindObject(new Guid(toComponent.InstanceId), false) as GH_Component;
            if (toGhComponent is null) { return; }
            var toParam = toGhComponent.Params.Input.Find(o => o.Name.ToLower().Contains(pairing.ToInputName.ToLower()));
            if (toParam is null)
            {
                toParam = toGhComponent.Params.Output[0];
            }

            var fromObject = doc.FindObject(new Guid(fromComponent.InstanceId), false);
            if (fromObject is null) { return; }
            IGH_Param fromParam;
            if (fromObject is GH_NumberSlider || fromObject is GH_Panel)
            {
                fromParam = fromObject as IGH_Param;

            }
            else
            {
                var fromGhComponent = fromObject as GH_Component;
                fromParam = fromGhComponent.Params.Output.Find(o => o.Name.ToLower().Contains(pairing.FromOutputName.ToLower()));
                if (fromParam is null)
                {
                    fromParam = fromGhComponent.Params.Output[0];
                }
            }


            toParam.AddSource(fromParam);
            toParam.CollectData();
            toParam.ComputeData();

        }

        public static void InstantiateComponent(GH_Document doc, GH_Component component, Component addition)
        {
            try
            {
                string name = addition.Name;
                IGH_ObjectProxy myProxy = GetObject(addition);
                if (myProxy is null)
                    return;

                var emit = Instances.ComponentServer.EmitObject(myProxy.Guid);
                addition.InstanceId = emit.InstanceGuid.ToString();
                doc.AddObject(emit, false);

                var thisPivot = component.Attributes.Pivot;
                emit.Attributes.Pivot = new System.Drawing.PointF(x: (float)addition.Position.X + thisPivot.X, y: (float)addition.Position.Y + thisPivot.Y);
            }
            catch
            {
            }
        }
    }
}