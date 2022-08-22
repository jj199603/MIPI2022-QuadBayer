
##  How to evaluate the IQ metrics 
`evaluate.py` - is to get the IQ metrics between the submitted bayer and the gt bayer
```commandline
    python ./program/evaluate.py <input_dir> <output_dir>

```
| Path                                                                                                 | Format | Description        | 
|:-----------------------------------------------------------------------------------------------------|-------:|:-------------------| 
| &#9500;&#9472;&nbsp; input                                                                           |              
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&nbsp; res                                         |   .bin | submitted bayer    |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&nbsp; ref                                         |        |                    |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&nbsp; img     |   .bin | ground truth bayer |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9500;&#9472;&nbsp; imgInfo |   .xml | image information  |

1. The submitted results in `/res` should be bayers of 10 bits in .bin format 
2. The ground truth in `/ref/img` should be bayers of 10 bits in .bin format 
3. The image info in `/ref/imgInfo` should be in the xml format


