import React, { useContext, useState } from "react";
import { TouchableOpacity, StyleSheet, Text, TextInput, View } from "react-native";
import { MaterialIcons } from "@expo/vector-icons";
import ChatHistContext from "../_context/ChathistContext";

export default function Chatbox() {
    const [query, setQuery] = useState("What is the historian's role in using myths?");
    const { askQuery } = useContext(ChatHistContext)
    return (
        <View style={styles.viewStyles}>
            <TextInput value={query} onChangeText={setQuery} style={styles.inputStyle} placeholder="How does a historian evaluate events?" placeholderTextColor='#94a3b8' />
            <TouchableOpacity
                onPress={() => {
                    askQuery(query)
                    setQuery('');
                }
                }
                style={
                    {
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        width: '15%'
                    }}>
                <MaterialIcons name='arrow-forward' color='#475569' size={30} />
            </TouchableOpacity>
        </View>
    );
}

const styles = StyleSheet.create({
    viewStyles: {
        height: 60,
        display: 'flex',
        flexDirection: 'row',
        borderTopLeftRadius: 20,
        borderTopRightRadius: 20,
        backgroundColor: 'white'
    },
    inputStyle: {
        flex: 1,
        paddingLeft: 10,
        outlineWidth: 0,
    }
});
