using System;

namespace Preprocess.Model
{
    public class Parameter
    {
        public string Id { get; set; }
        public string InstanceId { get; set; }
        public int Index { get; set; }
        public ParameterType ParameterType { get; set; }
        public Type DataType { get; set; }
        public string Name { get; set; }
        public string Nickname { get; set; }
        public string Description { get; set; }
        public string[] ConnectedParameterIds { get; set; }
        public string[] ConnectedInstanceIds { get; set; }
        public int[] ConnectedIndeces { get; set; }

    }


    public enum ParameterType
    {
        Input,
        Output
    }
}
