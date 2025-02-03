using System.ComponentModel;
using Newtonsoft.Json;

namespace Codesigner.Models
{
    public class Position
    {
        [JsonProperty("x")]
        [Description("The x-coordinate on the canvas.")]
        public double X { get; set; }

        [JsonProperty("y")]
        [Description("The y-coordinate on the canvas.")]
        public double Y { get; set; }
    }
}