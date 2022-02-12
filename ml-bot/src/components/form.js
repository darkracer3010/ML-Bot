
import React, { useState } from "react";
import axios from "axios";
import '../public/form.css'
const Form = ()=>{
    const [userName, setuserName] = useState('')
    const [noOfAttr, setnoOfAttr] = useState(0)
    const [typeofLearning, settypeofLearning] = useState('')
    
    const data={
        name:userName,
        attr:noOfAttr,
        learning:typeofLearning,
        
    };

    const SubmitData=async (e)=>{
        await axios.post("http://localhost:{port}/api/",data);
        console.log("send")
    }
    
    return (
        <div className="container box1 box2 background">
              <div className="box">
                    <label>User Name</label>
                    <input
                    type="text"
                    id="userName"
                    name="username"
                    placeholder="Enter your Name here"
                    onChange={(e)=>{setuserName(e.target.value)}}
                    ></input>

                    <label>No of Attributes Used</label>
                    <input
                    type="text"
                    id="noattr"
                    name="noattr"
                    placeholder="No of Attr here.."
                    onChange={(e)=>{setnoOfAttr(e.target.value)}}
                    ></input>
  
                    <label>Type of Learning Required</label>
                    <select id="learning" name="learning" onChange={(e)=>{settypeofLearning(e.target.value)}}>
                    <option selected value="Select One" disabled>
                          Select One
                    </option>
                    <option value="Supervised">Supervised</option>
                    <option value="Un-Supervised">Un-Supervised</option>
                    <option value="Semi Supervised">Semi Supervised</option>
                    
                    </select>
  
                    <label>Description About the Data</label>
                    <textarea
                    name="description"
                    id='description'
                    cols="25"
                    rows="4"
                    placeholder="Tell About the data used and purpose.."
                    
                    ></textarea>
  
                    <label>Upload the File Here</label>
                    <input
                    type="file"
                    id="data"
                    name="data"
                    
                    ></input>
            <br></br>
                    <input type="submit" value="Submit" onClick={SubmitData}></input>
              </div>
        </div>
        );

}
export default Form;