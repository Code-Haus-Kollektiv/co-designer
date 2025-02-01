using System;
using System.Drawing;
using Grasshopper;
using Grasshopper.Kernel;

namespace OpenAI
{
  public class OpenAIInfo : GH_AssemblyInfo
  {
    public override string Name => "Co-designer";

    //Return a 24x24 pixel bitmap to represent this GHA library.
    public override Bitmap Icon => null;

    //Return a short string describing the purpose of this GHA library.
    public override string Description => "Generate Grasshopper scripts using OpenAI";

    public override Guid Id => new Guid("cfdda7ea-d308-4d6a-8d5a-5ea670a89679");

    //Return a string identifying you or your company.
    public override string AuthorName => "";

    //Return a string representing your preferred contact details.
    public override string AuthorContact => "";

    //Return a string representing the version.  This returns the same version as the assembly.
    public override string AssemblyVersion => GetType().Assembly.GetName().Version.ToString();
  }
}