import { StyleSheet, Text, View } from "react-native";
import React, { useContext } from "react";
import ChatHistContext from "@/app/_context/ChathistContext";
import Chat from "./Chat";

export default function Content() {
    const { messages } = useContext(ChatHistContext)
    return messages.map((msg, index) => {
        return <Chat key={index} role={msg.role}>
            <Text style={styles.roleStyle}>{msg.role}</Text>
            <Text style={styles.contentStyle}>{msg.content}</Text>
        </Chat>
    })
}

const styles = StyleSheet.create({
    contentStyle: {
        fontSize: 15,
    },
    roleStyle: {
        fontSize: 15,
        fontWeight: 'bold',
        paddingVertical: 5,
        textTransform: 'capitalize'
    }
});
