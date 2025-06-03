import { StyleSheet, Text, View } from "react-native";
import React, { useContext } from "react";
import ChatHistContext from "@/app/_context/ChathistContext";
import Chat from "./Chat";
import { ScrollView } from "react-native-gesture-handler";

export default function Content() {
    const { messages } = useContext(ChatHistContext)
    return <ScrollView>
        {messages.map((msg, index) => {
            return <Chat key={index} role={msg.role}>
                <Text style={styles.roleStyle}>{msg.role}</Text>
                <Text style={styles.contentStyle}>{msg.content}</Text>
            </Chat>
        })}
    </ScrollView>
}

const styles = StyleSheet.create({
    contentStyle: {
        fontSize: 15,
        lineHeight: 25,
        letterSpacing: 0.5
    },
    roleStyle: {
        fontSize: 17,
        fontWeight: 'bold',
        paddingVertical: 5,
        textTransform: 'capitalize'
    }
});
